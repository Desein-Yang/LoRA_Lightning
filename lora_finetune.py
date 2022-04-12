import pdb, os
from sched import scheduler
import torch
from loralib.utils import lora_state_dict
from loralib.layers import Linear
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from data.data_loader import GLUEDataLoader

from transformers import BertModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import loralib as lora
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class LoraBertFinetuner(pl.LightningModule):
    def __init__(self, args, dataset=None):
        super(LoraBertFinetuner, self).__init__()
        # do not automaticly optimize by grad
        self.automatic_optimization = False

        # params about finetune
        self.lr = args.lr
        self.warmup_ratio = args.warmup_ratio
        self.args = args

        # model arch
        self.bert = BertModel.from_pretrained(
            "bert-base-cased", output_attentions=True
        )
        self.base_model = self.bert.base_model
        self.config = self.bert.config

        for p in self.base_model.parameters():
            p.requires_grad = False
        
        # params about lora
        self.apply_lora = args.lora
        self.lora_path = args.lora_path
        if self.apply_lora:
            self.apply_lora_linear(args.r, args.alpha)
            self.save_lora_params()
            self.add_trainable_params(args.lora_path)
            print("add lora layers")

            # if use manual opt
            self.automatic_optimization = False

        # classifier (2 classes)
        self.num_classes = 2
        self.W = nn.Linear(self.bert.config.hidden_size, 2)

        # load dataset
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = GLUEDataLoader(
                task=args.task, 
                train_batch_size=args.batch,
                max_seq_length=args.max_seq_length,
            )
        print("Config Dataset")

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
        #pdb.set_trace()
        # print keys of lora state dict
        print("Lora state dict keys" + str(lora_state_dict(self.bert.base_model).keys()))

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
                print("Replace " + name + " with lora embedding")

    def get_module_list(self):
        layer_names_dict = self.base_model.state_dict().keys()
        module_list = []

        for key in layer_names_dict:
            module_list.append('.'.join(key.split('.')[:-1]))
        
        return module_list

    # deprecated 
    def add_trainable_params(self, lora_path):
        model = self.bert.base_model

        for name, param in model.named_parameters():
            if "bert" or "roberta" in name:
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.require_grad = False
            else:
                param.requires_grad = True
        
        #print("Add " + name + " as trainable params")

        # trainable_params = []
        # # add lora as trainable params
        # if self.apply_lora:
        #     if self.lora_path:
        #         lora_state_dict = torch.load(lora_path)
        #         print("Load lora params from " + lora_path)
        #         print(lora_state_dict.keys())
        #         model.load_state_dict(lora_state_dict,strict=False)
        
        #     trainable_params.append('lora')
        # # add bit or adapter as trainable params
        # # do not use now

        # # set required_grad
        # if len(trainable_params) > 0:
        #     for name, param in model.named_parameters():
        #         for trainable_param in trainable_params:
        #             if "lora_" in name:
        #                 param.requires_grad = True
        #                 print("Add " + name + " as trainable params")

        assert self.sum_trainable_params() != 0
        print("Sum of trainable params: " + str(self.sum_trainable_params()))

    def sum_trainable_params(self):
        return sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)

    def save_lora_params(self,checkpoint_path=None):
        model = self.bert.base_model
        if checkpoint_path is None:
            checkpoint_path = self.lora_path

        lora_dict = lora_state_dict(model)
        print(lora_dict.keys())
        torch.save(lora_dict, checkpoint_path)
        print("Save lora params to " + checkpoint_path)
        print("Lora dict "+str(lora_dict.keys()))

    def get_flatten_params(self):
        state_dict = self.bert.base_model.state_dict()
        flatten_params = torch.tensor([])
        for key in state_dict.keys():
            torch.cat([
                flatten_params,
                torch.flatten(state_dict[key])
            ])
        return flatten_params

    def log_opt(self):

        if self.last_params:
            self.curr_params = self.get_flatten_params()
            for l,c in zip(self.last_params,self.curr_params):
                delta = torch.sub(c,l)

            self.log_dict({
                    "delta_max": delta.max(),
                    "delta_min": delta.min(),
                    "delta_mean": delta.mean(),
                    "delta_sum": delta.sum()
                })
            
            self.last_params = self.curr_params
                     
        else:
            self.last_params = self.get_flatten_params()

    def forward(self, input_ids, attention_mask, token_type_ids):
        h_cls = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0][:, 0]
        attn = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[2]
        logits = self.W(h_cls)

        # output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return logits, attn

    # Manual Optimization
    # torch lightning don't use the train eval in loralib/layer (why?)
    # replace training step
    def training_step(self, batch, batch_nb):
        opt = self.optimizers()
        opt.zero_grad()

        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        # loss.requires_grad_(True)
        
        loss.backward(create_graph=True)
        # self.manual_backward(loss)
    
        opt.step()

        tensorboard_logs = {"train_loss": loss}
        self.log("loss",loss, logger=True, prog_bar=True)
        
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], logger=True, prog_bar=True)
        
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

    def configure_optimizers(self):
        def get_total_opt_steps():
            tb_size = self.train_dataloader().batch_size * max(1, self.args.gpus)
            step_per_epoch =  len(self.train_dataloader().dataset) // tb_size 
            return self.args.epoch * step_per_epoch

        print("Config Optimizer Adam with params:")
        # need not to change now (lora params also use adam)
        num_trainable_params = sum([p.numel() for p in self.parameters() if p.requires_grad]) / 1e6
        print("Num trainable params: " + str(num_trainable_params))
        

        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08
        )
        self.last_params = None

        # Add warmup scheduler Lora
        self.total_steps = get_total_opt_steps()
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)

        self.warmup_steps = 0
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]
        
    # # function hook in LightningModule
    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     optimizer.step(closure=optimizer_closure) 
    #     print("optimizer steps")

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_accu", mode="max")
        ckpt = ModelCheckpoint(
            self.logger[0].log_dir, save_top_k=-1, verbose=True, 
            save_on_train_epoch_end=True, 
            monitor='val_accu', mode='max'
        )
        return [early_stop,ckpt]

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     optimizer.step(closure=optimizer_closure)

    # @pl.data_loader
    def train_dataloader(self):
        return self.dataset.train_dataloader()

    # @pl.data_loader
    def val_dataloader(self):
        return self.dataset.val_dataloader()

    # @pl.data_loader
    def test_dataloader(self):
        return self.dataset.test_dataloader()
