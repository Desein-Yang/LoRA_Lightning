import pdb
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from data.data_loader import GLUEDataLoader

# from transformers import BertModel
from model import LoraBertModel, LoraBertConfig
from torch.nn import Linear
import torch.nn.functional as F


class LoraBertFinetuner(pl.LightningModule):
    def __init__(self, args, dataset=None):
        super(LoraBertFinetuner, self).__init__()
        # do not automaticly optimize by grad
        self.automatic_optimization = False

        # params about finetune
        self.lr = args.lr

        # params about lora
        self.apply_lora = args.lora
        if self.apply_lora:
            self.lora_r = args.lora_r
            self.lora_alpha = args.lora_alpha
            self.lora_path = args.lora_path
            print("Load params from " + self.lora_path)

        # model arch
        self.bert = LoraBertModel.from_pretrained(
            "bert-base-cased", output_attentions=True
        )
        # classifier (2 classes)
        self.num_classes = 2
        self.W = Linear(self.bert.config.hidden_size, 2)

        # load dataset
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = GLUEDataLoader(
                task=args.task, train_batch_size=args.batch_size
            )
        print("Config Dataset")

    # TODO: add lora layer (BertModel Class)
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

    # TODO: modify optimizer
    def training_step(self, batch, batch_nb):
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
        print("Config Optimizer Adam")
        # TODO: config yourself
        # TODO: need not to change now (lora params also use adam)
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08
        )

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
