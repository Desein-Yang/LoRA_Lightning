
import pytorch_lightning as pl
from finetune import BertFinetuner,BertModel2
from pytorch_lightning import Trainer
from argparse import ArgumentParser

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

parser = ArgumentParser()
#parser = Trainer.add_argparse_args(parser)
parser.add_argument("--run_id", default="runs", help="runs name",type=str)
parser.add_argument("--task", default='sst2', help = "task",type=str)
parser.add_argument("--batch", default = 16, help = "batch size", type=int)
parser.add_argument("--lr", default = 5e-4, help = "learning rate", type=float)
parser.add_argument("--epoch", default = 60, help = "epoch",type=int)
parser.add_argument("--gpus", default = 1, help = "gpus",type=int)
args = parser.parse_args()

bert_finetuner = BertFinetuner(task=args.task, batch_size=args.batch, lr=args.lr)
model = BertModel2()

logdir = "bert-ft/"
tb_logger = TensorBoardLogger(logdir,name=args.run_id)
csv_logger = CSVLogger(logdir,name=args.run_id)

trainer = pl.Trainer(gpus=args.gpus, max_epochs = args.epoch, default_root_dir=logdir, logger=[tb_logger,csv_logger])
trainer.fit(bert_finetuner)
trainer.save_checkpoint(logdir+'/'+args.run_id+"/last_model.ckpt")
trainer.test()
#trainer.test(model,ckpt_path=logdir+"/last_model.ckpt")