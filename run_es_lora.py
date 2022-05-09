import pytorch_lightning as pl
from finetuner import EaLoraBertFinetuner
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from configparser import ConfigParser
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from optim.utils import setup, cleanup
import sys, os, logging, traceback

def load_config(from_config=True,config_name='config.ini'):
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--run_id", default="runs", help="runs name", type=str)
    parser.add_argument("--task", default="sst2", help="task", type=str)
    parser.add_argument("--model_id", default="bert-base-cased", help="task", type=str)
    parser.add_argument("--batch", default=32, help="batch size", type=int)
    parser.add_argument("--lr", default=2e-5, help="learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.01, help="L2 regularization", type=float)
    parser.add_argument("--warmup_ratio", default=0.06, help="warmup learning rate", type=float)
    parser.add_argument("--warmup_steps", default=0, help="warmup learning rate", type=int)
    parser.add_argument("--max_updates", default=0, help="max update", type=int)
    parser.add_argument("--epoch", default=3, help="epoch", type=int)
    parser.add_argument("--gpus", default=1, help="gpus", type=int)
    parser.add_argument("--max_seq_length", default=128, help="lora path",type=int)
    parser.add_argument("--early_stop", default=0, help="lora path",type=int)
    parser.add_argument("--model_check", default=0, help="lora path",type=int)
    parser.add_argument("--test_only", default=0, help="lora path",type=int)
    parser.add_argument("--log_dir", default='./logs/', help="log path",type=str)
    # es params
    parser.add_argument("--optim", default='ea', help="es patience",type=str)    
    parser.add_argument("--use_select", default=0, help="if use select(CES) in ea",type=int)
    parser.add_argument("--use_ddp", default=1, help="if use dist",type=int)
    parser.add_argument("--sigma", default=1, help="sigma init in ea",type=int)
    parser.add_argument("--clip_l", default=-5, help="es patience",type=int)
    parser.add_argument("--clip_h", default=5, help="es patience",type=int)

    # because ea must use parallel
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)

    # lora params
    parser.add_argument("--lora", default=0, help="lora", type=int)
    parser.add_argument("--alpha", default=0.1, help="alpha", type=float)
    parser.add_argument("--r", default=1, help="r", type=int)
    parser.add_argument("--lora_path", default='./logs/', help="lora path",type=str)

    # only when you can't use command line(overwrite cmd)
    args = parser.parse_args()
    if from_config:
        config = ConfigParser()
        config.read('./config/'+config_name)
        args_list = []
        for k,v in config['train'].items():
            type_func = args.__dict__[k].__class__
            name = k
            logging.info(name,":",type_func)
            args.__setattr__(k,type_func(v))

    return args

def run_dist(func, args):
    setup(args.local_rank, args.local_world_size, init_methods=None)
    try: 
        func(args)
    except Exception as e:
        logging.error("Main program error:")
        logging.error(e)
        logging.error(traceback.format_exc())     
        cleanup()

        
def main(args):
    finetuner = EaLoraBertFinetuner(args)

    tb_logger = TensorBoardLogger(args.log_dir, name=args.run_id)
    csv_logger = CSVLogger(args.log_dir, name=args.run_id)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epoch,
        default_root_dir=args.log_dir,
        logger=[tb_logger, csv_logger],
    )
    if not args.test_only:
        trainer.fit(finetuner)
        trainer.test(ckpt_path='best', verbose=True)
        if args.lora:
            finetuner.save_lora_params(checkpoint_path="./logs/ea-lora-ft/lora_params_last.ckpt")
    else: 
        trainer.test(ckpt_path='best', verbose=True)

if __name__ == "__main__":
    logging.basicConfig(filename='traceback.log',
            level=logging.INFO, filemode='a', 
            format='[%(asctime)s] [%(levelname)s] >>>  %(message)s',
            datefmt='%Y-%m-%d %I:%M:%S')
    # True if run from python run_lora.py
    # Flase if run from bash run_lora.sh

    #args = load_config(True,'bert-config.ini')
    #args = load_config(True,'roberta-config.ini')
    args = load_config(True,'ealora-config.ini')
    logging.debug(args)

    if args.use_ddp:
        run_dist(main, args)
    else:
        main(args)
        
#with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#     bert_finetuner.train_dataloader()
#    outputs = bert_finetuner.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#    print(prof.table())
