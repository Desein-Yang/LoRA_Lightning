import imp
import pdb, os, sys, logging
import torch
from loralib.layers import Linear
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from data.data_loader import GLUEDataLoader

from transformers import BertModel, BertTokenizer, RobertaModel
import torch.nn as nn
import loralib as lora
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.distributed as dist

sys.path.append('..')
sys.path.append('.')
from .utils import idle_device, sum_trainable_params, warmup_fn
from .utils import sum_trainable_params, sum_grad_params
from .utils import flatten_tensor_dict, flatten_tensor_list
from optim.es import EvoStrategy
from optim.utils import sync_scalar
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class EaLoraBertFinetuner(pl.LightningModule):
    def __init__(self, args, dataset=None):
        super(EaLoraBertFinetuner, self).__init__()
       # params about finetune
        self.args = args

        # set ea optim and hyper
        self.automatic_optimization = False
        self.apply_ea = True if args.optim == 'ea' else False
        
        # set onto most free device
        self.rank =  dist.get_rank()
        self.size = dist.get_world_size()
        idle_devices = idle_device(self.size)
        torch.cuda.set_device(idle_devices[self.rank]) #device or int, need to be set  CUDA_VISIBLE_DEVICES 
        device = torch.cuda.current_device()
        self.log_rank(f"Device: {device}")
        
        # model arch
        self.bert = self.build_model().to(device)
        self.dataset = self.build_dataset(args, dataset)
        self.base_model = self.bert.base_model
        
        # set lora layer
        self.apply_lora = args.lora
        self.lora_path = args.lora_path
        if self.apply_lora:
            self.apply_lora_linear(args.r, args.alpha)
            self.save_lora_params()
            self.log_rank("add lora layers")
        
        # set trainable params before configure optimizer
        # self.trainable_params = self.set_trainable_params(self.base_model, self.apply_lora, args.apply_ea)

        # classifier (2 classes)
        self.num_classes = 2
        self.W = torch.nn.Linear(self.bert.config.hidden_size, 2)
        
        # load dataset
        self.dataset = self.build_dataset(args, dataset)
        
    def set_trainable_params(self, base_model, apply_lora=False, apply_ea=False):
        if apply_ea: 
            for p in base_model.parameters():
                p.requires_grad = False
            
            if apply_lora:
                trainable_params = [p for n,p in base_model.named_parameters() if 'lora_' in n] # lora + ea
                self.log_rank("Set Lora layer as trainable params without grad")
            else:
                trainable_params = [p for p in base_model.parameters()] # full finetune + ea
                self.log_rank("Set base model as trainable params without grad")
        else: 
            if apply_lora:
                for name, param in base_model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True # lora + adam
                self.log_rank("Set Lora layer as trainable params by grad") 
            else: 
                for p in base_model.parameters():
                    p.requires_grad = True # full finetune + adam
                self.log_rank("Set base model as trainable params by grad") 
            
            trainable_params = [p for p in base_model.parameters() if p.requires_grad] # full finetune or lora + adam 

        # summary
        num_trainable = sum_trainable_params(trainable_params)
        assert num_trainable != 0
        self.log_rank("Sum of trainable params: " + str(num_trainable))   
        
        return trainable_params

    def build_model(self):
        model_type = self.args.model_id.split('-')[0]
        if model_type == 'bert':
            model = BertModel.from_pretrained(self.args.model_id, hidden_dropout_prob=0.1, output_attentions=True)
        elif model_type == 'roberta':
            model = RobertaModel.from_pretrained(self.args.model_id, hidden_dropout_prob=0.1, output_attentions=True)
        self.log_rank(model.config)
        return model

    def build_dataset(self, args, dataset=None):
        self.log_rank("Config Dataset")
        if dataset is None:
            return GLUEDataLoader(
                task=args.task, 
                train_batch_size=args.batch,
                max_seq_length=args.max_seq_length,
            )
        else:
            return dataset

    # add lora layer by get_submodule
    def apply_lora_linear(self, r, alpha):
        # replace submodule_key in model with module 
        def _set_module(model, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = model
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)

        module_list = self.get_module_list()
        for submodule_key in module_list:
            # load the original state dict
            module_state_dict = self.base_model.get_submodule(submodule_key).state_dict()

            # source code of loralib only replace query and value
            if submodule_key.split('.')[-1] in ["query","value"]: # Linear should be replaced    
                submodule = self.base_model.get_submodule(submodule_key)
                lora_layer = lora.Linear(
                    submodule.in_features,
                    submodule.out_features,
                    r = r,
                    lora_alpha= alpha,
                    lora_dropout = 0.1
                )

                lora_layer.load_state_dict(module_state_dict,strict=False)
                # params reset in __init__

                for n,p in lora_layer.named_parameters():
                    if 'lora' in n:
                        p.requires_grad = True

                _set_module(self.base_model, submodule_key, lora_layer)
                #print("Replace " + submodule_key + " with lora linear")

    # add embedding by get_module
    def apply_lora_embedding(self):
        module_list =self.get_module_list()
        for name in module_list:
            if name.split('.')[1] == "embedding":
                submodule = self.base_model.get_submodule(name)
                num_embeddings, embedding_dim = submodule.num_embeddings,submodule.embedding_dim
                submodule = lora.Embedding(
                    num_embeddings,
                    embedding_dim,
                    r = self.lora_r,
                    lora_alpha = self.lora_alpha
                )
                self.log_rank("Replace " + name + " with lora embedding")

    def get_module_list(self):
        layer_names_dict = self.base_model.state_dict().keys()
        module_list = []

        for key in layer_names_dict:
            module_list.append('.'.join(key.split('.')[:-1]))
        
        return module_list

    @staticmethod
    def lora_state_dict(my_state_dict):
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}

    def save_lora_params(self,checkpoint_path=None):
        model = self.bert.base_model
        if checkpoint_path is None:       
            checkpoint_path = "./logs/lora-ft/lora_params_0.ckpt"

        lora_dict = self.lora_state_dict(model.state_dict())
        torch.save(lora_dict, checkpoint_path)
        self.log_rank("Save lora params to " + checkpoint_path)
        self.log_rank("Lora dict is "+str(lora_dict.keys()))

    def log_opt(self):
        if self.last_params is not None:
            self.curr_params = flatten_tensor_dict(self.base_model.state_dict()) 
            delta = torch.sub(self.curr_params,self.last_params)
            
            self.log_dict({
                    "delta_max": delta.max(),
                    "delta_min": delta.min(),
                    "delta_mean": delta.mean(),
                    "delta_sum": delta.sum()
                })
            
            self.last_params = self.curr_params                     
        else:
            self.last_params = flatten_tensor_dict(self.base_model.state_dict()) 

    def log_rank(self, msg):
        if self.rank == 0:
            logging.info(f"[{self.rank}] {msg}")

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        h_cls = output[0][:, 0]
        attn = output[2]
        logits = self.W(h_cls)

        return logits, attn

    # Manual Optimization
    # torch lightning don't use the train eval in loralib/layer (why?)
    # replace training step
    def training_step(self, batch, batch_nb):
        if self.apply_ea:
            self.training_step_nograd(batch, batch_nb) # ea
        else:
            self.training_step_grad(batch, batch_nb) # adamw
    
    def training_step_nograd(self, batch, batch_nb):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.mutate()
        opt.log_params("Mutate Params:")

        input_ids, attention_mask, token_type_ids, label = batch
        with torch.no_grad():
            y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        loss = F.cross_entropy(y_hat, label)
        seed = torch.initial_seed()

        sync_loss = sync_scalar(loss, opt.size)
        sync_seed = sync_scalar(seed, opt.size)
        
        self.log_rank(f"Sync_loss    : {sync_loss}")
        self.log_rank(f"Sync_seed    : {sync_seed}")

        opt.log_params("Before Update:")
        opt.step(closure=None, loss=sync_loss, seed=sync_seed)
        opt.log_params("After  Update:")

        tensorboard_logs = {"train_loss": loss}
        self.log("loss",loss, logger=True, prog_bar=True)
        self.log("lr", sch.get_last_lr()[0], logger=True, prog_bar=True)
        
        return {"loss": loss.detach(), "log": tensorboard_logs} 

    def training_step_grad(self, batch, batch_nb):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        
        loss.backward(create_graph=True)

        sch.step()
        opt.step()

        tensorboard_logs = {"train_loss": loss}
        self.log("loss",loss, logger=True, prog_bar=True)
        
        self.log("lr", sch.get_last_lr()[0], logger=True, prog_bar=True)
        
        self.log_opt()

        return {"loss": loss.detach(), "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(y_hat, label)

        a, y_hat = torch.max(y_hat, dim=1)
        # loss = torch.nn.BCELoss(y_hat,label)

        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        # log
        self.log("val_loss", loss, logger=True)
        self.log("val_accu", val_acc, logger=True)
        return {"val_loss": loss, "val_acc": val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        # log
        tensorboard_logs = {"val_loss": avg_loss, "avg_val_acc": avg_val_acc}
        self.log("val_loss", avg_loss, logger=True)
        self.log("avg_val_acc", avg_val_acc, logger=True)
        return {"avg_val_loss": avg_loss, "progress_bar": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        # log
        self.log("test_acc", test_acc, logger=True)
        return {"test_acc": torch.tensor(test_acc)}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        # log
        tensorboard_logs = {"avg_test_acc": avg_test_acc}
        self.log("avg_test_acc", avg_test_acc, logger=True)
        return {
            "avg_test_acc": avg_test_acc,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def get_total_opt_steps(self):
        if self.args.max_updates == 0:
            step_per_epoch =  len(self.train_dataloader().dataset) // self.train_dataloader().batch_size
            self.log_rank("step per epoch: "+str(step_per_epoch))
            return self.args.epoch * step_per_epoch
        else: 
            return self.args.max_updates

    def get_scheduler(self, optimizer, type='linear'):
        # Add warmup scheduler Lora
        self.total_steps = self.get_total_opt_steps()
        self.warmup_steps = self.args.warmup_steps
        if self.warmup_steps == 0:
            self.warmup_steps = int(self.total_steps * self.args.warmup_ratio)
        
        self.log_rank("Lr warmup steps:" + str(self.warmup_steps))
        self.log_rank("Total steps:" + str(self.total_steps))

        # ref: transformers/ get_linear_schedule_with_warmup
        lr_lambda = warmup_fn(self.warmup_steps,self.total_steps, type="linear")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer=optimizer,
                        lr_lambda=lr_lambda,
                        last_epoch=-1,
            )    
        scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        self.log_rank(f"Lr Scheduler {type}: frequency 1, interval step")
        return scheduler

    def configure_optimizers(self):
        # set trainable params
        trainable_params = self.set_trainable_params(self.base_model, self.apply_lora, self.apply_ea)
        num_trainable_params = sum_trainable_params(trainable_params) / 1e6
        self.log_rank("Num trainable params: " + str(num_trainable_params))
        
        if self.apply_ea:
            self.log_rank("Configuring optimizer with EA")
            optimizer = EvoStrategy(
                trainable_params, 
                lr=self.args.lr, 
                #weight_decay=self.args.weight_decay,
                #sigma=self.args.sigma_init,
                select=self.args.use_select,
                #clip=(self.args.clip_l,self.args.clip_h),
                #verbose=self.args.verbose,
                eps = 1e-8,
            )
            self.last_params = None
        else:
            self.log_rank("Config Optimizer AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.lr, 
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
            self.last_params = None

        # set scheduler
        scheduler = self.get_scheduler(optimizer, type='linear')
        return [optimizer], [scheduler]
        
    def configure_callbacks(self):
        callbacks = []
        if self.args.early_stop:
            early_stop = EarlyStopping(monitor="val_accu", mode="max")
            callbacks.append(early_stop)
        if self.args.model_check:
            ckpt = ModelCheckpoint(
                dirpath = self.logger[0].sub_dir,
                filename="{epoch}-{val_loss:.2f}", 
                save_top_k=-1, verbose=True, 
                save_on_train_epoch_end=True, 
                monitor='val_accu', mode='max'
            )
            callbacks.append(ckpt)
        return callbacks

    # @pl.data_loader
    def train_dataloader(self):
        return self.dataset.train_dataloader()

    # @pl.data_loader
    def val_dataloader(self):
        return self.dataset.val_dataloader()

    # @pl.data_loader
    def test_dataloader(self):
        return self.dataset.test_dataloader()
