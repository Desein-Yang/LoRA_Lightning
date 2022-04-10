import pdb, os
#from sched import scheduler
import time
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from data.data_loader import GLUEDataLoader
from transformers import BertModel, BertConfig
from transformers import RobertaModel, RobertaConfig, get_linear_schedule_with_warmup
from torch.nn import Linear, Dropout
import torch.nn.functional as F


class BertFinetuner(pl.LightningModule):
    def __init__(self, task="sst2", batch_size=32, lr=2e-5, warmup_steps = 10000, dataset=None):
        super(BertFinetuner, self).__init__()
        # use pretrained BERT
        self.bert = BertModel.from_pretrained(
            "bert-base-cased", hidden_dropout_prob=0.1, output_attentions=True
        )
        config = BertConfig.from_pretrained("bert-base-cased")
        print(config)

        # classifier (2 classes)
        self.num_classes = 2
        self.W = Linear(self.bert.config.hidden_size, 2)
        # self.Dropout = Dropout(p=0.1)
        self.lr = lr
        self.warmup_steps = warmup_steps # for 1e5 steps warmup lr linearly

        if dataset:
            self.dataset = dataset
        else:
            self.dataset = GLUEDataLoader(task=task, train_batch_size=batch_size)
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

        # if is_train:
        #    logits = self.Dropout(h_cls)
        logits = self.W(h_cls)

        # output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return logits, attn

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(
            input_ids, attention_mask, token_type_ids, is_train=True
        )

        # one-hot
        loss = F.cross_entropy(y_hat, label)
        # y_hat = torch.sigmoid(y_hat)
        # loss = torch.nn.BCELoss(y_hat,label)
        # logs
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
        print("Config Optimizer Adam")
        print("lr = {}".format(self.lr) + " warmup_steps = {}".format(self.warmup_steps))
        # pretrain weight decay = 0.1
        # fintune weight decay None
        def get_total_opt_steps():
            tb_size = 16 
            step_per_epoch =  len(self.train_dataloader().dataset) // tb_size
            return 10 * step_per_epoch

        # pretrain weight decay = 0.1
        # fintune weight decay None
        optimizer =  torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )

        # Add warmup scheduler Lora
        self.total_steps = get_total_opt_steps()
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


class RoBertaFinetuner(pl.LightningModule):
    def __init__(self, task="sst2", batch_size=32, lr=2e-5, warmup_ratio = 0.06, dataset=None):
        super(RoBertaFinetuner, self).__init__()
        # use pretrained BERT
        self.bert = RobertaModel.from_pretrained(
            "roberta-base", hidden_dropout_prob=0.1, output_attentions=True
        )
        config = RobertaConfig.from_pretrained("roberta-base")
        print(config)

        # classifier (2 classes)
        self.num_classes = 2
        self.W = Linear(self.bert.config.hidden_size, 2)
        # self.Dropout = Dropout(p=0.1)
        self.lr = lr
        self.warmup_ratio = warmup_ratio # first 10000 steps warmup to peak

        if dataset:
            self.dataset = dataset
        else:
            self.dataset = GLUEDataLoader(task=task, train_batch_size=batch_size)
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

        # if is_train:
        #    logits = self.Dropout(h_cls)
        logits = self.W(h_cls)

        # output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return logits, attn

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(
            input_ids, attention_mask, token_type_ids, is_train=True
        )
        # one-hot
        loss = F.cross_entropy(y_hat, label)

        # logs
        tensorboard_logs = {"train_loss": loss}
        self.log("loss", loss, logger=True, prog_bar=True)

        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], logger=True, prog_bar=True)

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
            tb_size = 16 
            step_per_epoch =  len(self.train_dataloader().dataset) // tb_size
            return 10 * step_per_epoch

        print("Config Optimizer Adam")
        # pretrain weight decay = 0.1
        # fintune weight decay None
        optimizer =  torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
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

    # def on_save_checkpoint(self, checkpoint):
    #     # 99% of use cases you don't need to implement this method
    #     checkpoint['epoch'] = self.current_epoch

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
