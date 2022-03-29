from ast import Try
import pdb
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from data.data_loader import GLUEDataLoader
from transformers import BertModel, BertTokenizer
from torch.nn import Linear
import torch.nn.functional as F


class BertFinetuner(pl.LightningModule):
    def __init__(self,task='sst2',batch_size=16,lr=5e-4,dataset=None):
        super(BertFinetuner, self).__init__()
        # use pretrained BERT        
        self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
        # classifier (2 classes)     
        self.num_classes = 2  
        self.W = Linear(self.bert.config.hidden_size, 2)       
        self.lr = lr 

        if dataset:
            self.dataset = dataset
        else:
            self.dataset = GLUEDataLoader(task='sst2',train_batch_size=batch_size)
        
        print('Config Dataset')  

    def forward(self, input_ids, attention_mask, token_type_ids):
        h_cls = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0][:, 0]   
        attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[2]      
        logits = self.W(h_cls)    
        #output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return logits, attn

    def training_step(self, batch, batch_nb):        
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        # one-hot
        loss = F.cross_entropy(y_hat, label)
        #y_hat = torch.sigmoid(y_hat)
        #loss = torch.nn.BCELoss(y_hat,label)
        # logs        
        tensorboard_logs = {'train_loss': loss}       
        self.log('loss',loss,logger=True,prog_bar=True)
        return {'loss': loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_nb):        
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        loss = F.cross_entropy(y_hat, label)

        a, y_hat = torch.max(y_hat, dim=1)       
        #loss = torch.nn.BCELoss(y_hat,label)

        val_acc = accuracy_score(y_hat.cpu(), label.cpu())        
        val_acc = torch.tensor(val_acc)
        
        # log
        self.log('val_loss',loss, logger=True)
        self.log('val_accu',val_acc, logger=True)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_end(self, outputs):        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()        
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        # log
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}        
        self.log('val_loss',avg_loss,logger=True)
        self.log('avg_val_acc',avg_val_acc,logger=True)
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):        
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        a, y_hat = torch.max(y_hat, dim=1)        
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        # log
        self.log('test_acc', test_acc, logger=True)
        return {'test_acc': torch.tensor(test_acc)}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        # log
        tensorboard_logs = {'avg_test_acc': avg_test_acc}        
        self.log('avg_test_acc',avg_test_acc, logger=True)
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):      
        print('Config Optimizer Adam')  
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    #@pl.data_loader    
    def train_dataloader(self):  
    #    print(type(self.dataset.train_dataloader()))
        return self.dataset.train_dataloader()

    #@pl.data_loader    
    def val_dataloader(self):        
       return self.dataset.val_dataloader()

    #@pl.data_loader    
    def test_dataloader(self):        
       return self.dataset.test_dataloader()

class BertModel2(pl.LightningModule):
    def __init__(self):
        super(BertModel2, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
        # classifier (2 classes)     
        self.num_classes = 2  
        self.W = Linear(self.bert.config.hidden_size, 2)  

        self.dataset = GLUEDataLoader(task='sst2')
        
        print('Config Dataset')  

    def forward(self, input_ids, attention_mask, token_type_ids):
        h_cls = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0][:, 0]   
        attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[2]      
        logits = self.W(h_cls)    
        #output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return logits, attn

    def test_dataloader(self):        
       return self.dataset.test_dataloader()

    def test_step(self, batch, batch_nb):        
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        a, y_hat = torch.max(y_hat, dim=1)        
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        # log
        self.log('test_acc', test_acc, logger=True)
        return {'test_acc': torch.tensor(test_acc)}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        # log
        tensorboard_logs = {'avg_test_acc': avg_test_acc}        
        self.log('avg_test_acc',avg_test_acc, logger=True)
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
