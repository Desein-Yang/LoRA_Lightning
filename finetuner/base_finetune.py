import pdb, os
import time
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from data.data_loader import GLUEDataLoader
from transformers import BertModel, RobertaModel, RobertaConfig
from torch.nn import Linear
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class BertFinetuner(pl.LightningModule):
    def __init__(self, args, dataset=None):
        super(BertFinetuner, self).__init__()
               # params about finetune
        self.args = args
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps # for 1e5 steps warmup lr linearly
        self.warmup_ratio = args.warmup_ratio # for 0.06 steps warmup lr linearly
        self.model_id = args.model_id
        self.automatic_optimization = True
        
        # model arch
        model_type = self.model_id.split('-')[0]
        if model_type == 'bert':
            self.bert = BertModel.from_pretrained(self.model_id, hidden_dropout_prob=0.1, output_attentions=True)
        elif model_type == 'roberta':
            self.bert = RobertaModel.from_pretrained(self.model_id, hidden_dropout_prob=0.1, output_attentions=True)
        
        print(self.bert.config)

        # classifier (2 classes)
        self.num_classes = 2
        self.W = Linear(self.bert.config.hidden_size, 2)

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

    def forward(self, input_ids, attention_mask, token_type_ids, is_train=False):
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

        return logits, attn

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(
            input_ids, attention_mask, token_type_ids, is_train=True
        )

        # one-hot
        loss = F.cross_entropy(y_hat, label)

        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log("lr", scheduler.get_last_lr()[0], logger=True, prog_bar=True)

        # logs
        tensorboard_logs = {"train_loss": loss}
        self.log("loss", loss, logger=True, prog_bar=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(y_hat, label)

        a, y_hat = torch.max(y_hat, dim=1)
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
            # step_per_epoch =  len(self.train_dataloader().dataset)
            step_per_epoch = 128 * self.args.batch # step = 4096 = 128 * 32
            print("step per epoch: "+str(step_per_epoch))
            return self.args.epoch * step_per_epoch        

        num_trainable_params = sum([p.numel() for p in self.parameters() if p.requires_grad]) / 1e6
        print("Num trainable params = {} M".format(str(num_trainable_params)))

        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        print("Config Optimizer Adam")

        # Add warmup scheduler Lora
        self.total_steps = get_total_opt_steps()
        if self.warmup_steps == 0:
            self.warmup_steps = int(self.total_steps * self.warmup_ratio)
        print("Lr warmup steps = {}".format(str(self.warmup_steps)))
        print("Total steps = {}".format(str(self.total_steps)))

        # ref: transformers/ get_linear_schedule_with_warmup
        def warmup(warmup_steps, total_steps):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
                ) 
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lr_lambda,
                last_epoch=-1,
            )
            
        scheduler = warmup(self.warmup_steps, self.total_steps)      
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]
        
    def configure_callbacks(self):
        callbacks = []
        if self.args.early_stop:
            early_stop = EarlyStopping(monitor="val_accu", mode="max")
            callbacks.append(early_stop)
        if self.args.model_check:
            ckpt = ModelCheckpoint(
                dirpath = self.logger[0].sub_dir ,
                filename="{epoch}-{val_loss:.2f}", 
                save_top_k=-1, verbose=True, 
                save_on_train_epoch_end=True, 
                monitor='val_accu', mode='max'
            )
            callbacks.append(ckpt)
        return callbacks         

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

