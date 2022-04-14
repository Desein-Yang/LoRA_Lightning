import pytorch_lightning as pl
from finetuner import BertFinetuner
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from configparser import ConfigParser
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

def load_config(from_config=True,config_name='bert-config.ini'):
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--run_id", default="runs", help="runs name", type=str)
    parser.add_argument("--task", default="sst2", help="task", type=str)
    parser.add_argument("--model_id", default="bert-base-cased", help="task", type=str)
    parser.add_argument("--batch", default=32, help="batch size", type=int)
    parser.add_argument("--lr", default=2e-5, help="learning rate", type=float)
    parser.add_argument("--warmup_ratio", default=0.06, help="warmup learning rate", type=float)
    parser.add_argument("--warmup_steps", default=0, help="warmup learning rate", type=int)
    parser.add_argument("--epoch", default=3, help="epoch", type=int)
    parser.add_argument("--gpus", default=1, help="gpus", type=int)
    parser.add_argument("--log_dir", default=None, help="log path",type=str)
    parser.add_argument("--max_seq_length", default=128, help="lora path",type=int)
    parser.add_argument("--early_stop", default=False, help="lora path",type=bool)
    parser.add_argument("--model_check", default=False, help="lora path",type=bool)
    parser.add_argument("--test_only", default=False, help="lora path",type=bool)

    # only when you can't use command line(overwrite cmd)
    if from_config:
        config = ConfigParser()
        config.read('./config/'+config_name)
        args_list = []
        for k,v in config['train'].items():
            args_list.append('--'+k)
            args_list.append(v)

        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()
    
    return args

def main(args):
    # fintuner
    finetuner = BertFinetuner(args)
    # roberta_finetuner = RoBertaFinetuner(task=args.task, batch_size=args.batch, lr=args.lr, warmup_ratio=args.warmup_ratio)
    #bert_finetuner = BertFinetuner(task=args.task, batch_size=args.batch, lr=args.lr, warmup_steps=args.warmup_steps)
    #model = BertModel2()

    # logdir
    logdir = args.log_dir
    tb_logger = TensorBoardLogger(logdir, name=args.run_id)
    csv_logger = CSVLogger(logdir, name=args.run_id)

    # fit
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epoch,
        default_root_dir=logdir,
        logger=[tb_logger, csv_logger],
    )
    
    if not args.test_only:
        trainer.fit(finetuner)
        trainer.test(ckpt_path='best',num_workers=16) # trainer.test(model,ckpt_path=logdir+"/last_model.ckpt")
    else: 
        trainer.test(finetuner, ckpt_path='best',num_workers=16)

if __name__ == "__main__":
    # True if run from python run_lora.py
    # Flase if run from bash run_lora.sh
    args = load_config(True,'roberta-config.ini')

    main(args)

