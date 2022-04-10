import pytorch_lightning as pl
from finetune import BertFinetuner, RoBertaFinetuner
from pytorch_lightning import Trainer
from argparse import ArgumentParser

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# add arguments
parser = ArgumentParser()
# parser = Trainer.add_argparse_args(parser)
parser.add_argument("--run_id", default="runs", help="runs name", type=str)
parser.add_argument("--task", default="sst2", help="task", type=str)
parser.add_argument("--batch", default=32, help="batch size", type=int)
parser.add_argument("--lr", default=2e-5, help="learning rate", type=float)
parser.add_argument("--epoch", default=3, help="epoch", type=int)
parser.add_argument("--warmup_steps", default=1e5, help="gpus", type=int)
parser.add_argument("--warmup_ratio", default=0.06, help="gpus", type=float)
parser.add_argument("--gpus", default=1, help="gpus", type=int)
args = parser.parse_args()

# fintuner
roberta_finetuner = RoBertaFinetuner(task=args.task, batch_size=args.batch, lr=args.lr, warmup_ratio=args.warmup_ratio)
#bert_finetuner = BertFinetuner(task=args.task, batch_size=args.batch, lr=args.lr, warmup_steps=args.warmup_steps)
#model = BertModel2()

# logdir
logdir = "roberta-ft/"
tb_logger = TensorBoardLogger(logdir, name=args.run_id)
csv_logger = CSVLogger(logdir, name=args.run_id)
ckpt_callbacks = ModelCheckpoint(
    logdir, 
    file_name = '{epoch}',
    save_top_k=-1, verbose=True, 
    save_on_train_epoch_end=True, 
    monitor='val_accu', mode='max'
)

# fit
trainer = pl.Trainer(
    gpus=args.gpus,
    max_epochs=args.epoch,
    default_root_dir=logdir,
    logger=[tb_logger, csv_logger],
    callbacks=[ckpt_callbacks],
)
trainer.fit(roberta_finetuner)

# test
trainer.save_checkpoint(logdir + "/" + args.run_id + "/last_model.ckpt")
trainer.test()
# trainer.test(model,ckpt_path=logdir+"/last_model.ckpt")

