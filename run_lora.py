from gc import callbacks
import pytorch_lightning as pl
from lora_finetune import LoraBertFinetuner
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from configparser import ConfigParser
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from memory_profiler import profile

parser = ArgumentParser()
# parser = Trainer.add_argparse_args(parser)
parser.add_argument("--run_id", default="runs", help="runs name", type=str)
parser.add_argument("--task", default="sst2", help="task", type=str)
parser.add_argument("--batch", default=32, help="batch size", type=int)
parser.add_argument("--lr", default=2e-5, help="learning rate", type=float)
parser.add_argument("--warmup_ratio", default=0.06, help="warmup learning rate", type=float)
parser.add_argument("--epoch", default=3, help="epoch", type=int)
parser.add_argument("--gpus", default=1, help="gpus", type=int)
parser.add_argument("--lora", default=True, help="lora", type=bool)
parser.add_argument("--alpha", default=0.1, help="alpha", type=float)
parser.add_argument("--r", default=1, help="r", type=int)
parser.add_argument("--lora_path", default=None, help="lora path",type=str)
parser.add_argument("--max_seq_length", default=128, help="lora path",type=int)

from_config = True
if from_config:
    config = ConfigParser()
    config.read('./config/config.ini')
    args_list = []
    for k,v in config['train'].items():
        args_list.append('--'+k)
        args_list.append(v)

    args = parser.parse_args(args_list)
else:
    args = parser.parse_args()

@profile(precision=4, stream=open("memory_profiler.txt", "w+"))
def main():
    bert_finetuner = LoraBertFinetuner(args)

    logdir = "lora-ft/"
    tb_logger = TensorBoardLogger(logdir, name=args.run_id)
    csv_logger = CSVLogger(logdir, name=args.run_id)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epoch,
        default_root_dir=logdir,
        logger=[tb_logger, csv_logger],
    )
    trainer.fit(bert_finetuner)
    bert_finetuner.save_lora_params()
    # trainer.save_checkpoint(logdir + "/" + args.run_id + "/last_model.ckpt")
    trainer.test()
    # trainer.test(model,ckpt_path=logdir+"/last_model.ckpt")

if __name__ == "__main__":
    main()
    
#with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#     bert_finetuner.train_dataloader()
#    outputs = bert_finetuner.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#    print(prof.table())
