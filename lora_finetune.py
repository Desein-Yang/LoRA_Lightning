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

                # initial weight by lora
                lora_layer.reset_parameters()

                _set_module(self.base_model, submodule_key, lora_layer)
                
                print("Replace " + submodule_key + " with lora linear")
        #pdb.set_trace()
        # print keys of lora state dict
        print("Lora state dict keys" + str(lora_state_dict(self.bert.base_model).keys()))

    # add embedding by get_module
    # do not use now
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

    def add_trainable_params(self, lora_path):
        model = self.bert.base_model

        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                print("Add " + name + " as trainable params")
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

        if self.sum_trainable_params() == 0:
            print("No trainable params")
            pdb.set_trace()
        else:
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
        print("Lora dict"+str(lora_dict.keys()))

    def log_opt(self):
        opt = self.optimizers()
        #logdir = self.logger().logdir
        if self.last_params:
            self.curr_params = opt.param_groups[0]['params']
            for l,c in zip(self.last_params,self.curr_params):
                delta = torch.sub(c,l)

            self.log_dict({
                    "delta_max": delta.max(),
                    "delta_min": delta.min(),
                    "delta_mean": delta.mean(),
                    "delta_sum": delta.sum()
                })
                     
        else:
            self.last_params = opt.param_groups[0]['params']

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

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        
        tensorboard_logs = {"train_loss": loss}
        self.log("loss",loss, logger=True, prog_bar=True)
        
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], logger=True, prog_bar=True)

        return {"loss": loss.detach(), "log": tensorboard_logs}

    def on_train_epoch_end(self):
        self.log_opt()

    # TODO: deprecated modify optimizer
    def selfopt_training_step(self, batch, batch_nb):
        optim = self.optimizer()
        optim.zero_grad()

        # self.compute_loss(batch)
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)

        self.manual_backward(loss)

        # update lr every N epoch
        sch = self.lr_schedulers()
        if (batch_nb + 1) % N == 0:
            sch.step()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % N == 0:
            sch.step()

        tensorboard_logs = {"train_loss": loss}
        self.log("loss", loss, logger=True, prog_bar=True)
        return {"loss": loss, "log": tensorboard_logs}

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

        # Add warmup scheduler Lora
        self.total_steps = get_total_opt_steps()
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)
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

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_acc", mode="max")
        pdb.set_trace()
        ckpt = ModelCheckpoint(
            self.logger.log_dir, save_top_k=-1, verbose=True, 
            save_on_train_epoch_end=True, 
            monitor='val_accu', mode='max'
        )
        return [early_stop,ckpt]

    # @pl.data_loader
    def train_dataloader(self):
        #    print(type(self.dataset.train_dataloader()))
        return self.dataset.train_dataloader()

    # @pl.data_loader
    def val_dataloader(self):
        return self.dataset.val_dataloader()

    # @pl.data_loader
    def test_dataloader(self):
        return self.dataset.test_dataloader()
