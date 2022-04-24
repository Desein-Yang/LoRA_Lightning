from os import sync
import numpy as np
from regex import W
import torch
import torch.optim as optim
from torch.nn import Linear, MSELoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from optim.evolution import EA
import argparse

params1 = torch.Tensor([1, 2, 3])
params2 = torch.Tensor([1, 2, 3])
params = [params1, params2]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Linear(2,2)
        self.w2 = Linear(2,2)
    
    def forward(self, x):
        return self.w2(self.w1(x))

def log_params(optim):
    """Log parameters of optimizer with rank id."""
    rank = dist.get_rank()
    params = optim.param_groups[0]['params'][0]
    print(f"[{rank}]{params}")

def log(msg):
    """Return log message with rank id."""
    rank = dist.get_rank()
    print(f"[{rank}] {msg}")

def build_gauss(model: torch.nn.Module, sigma_init):
    """Build a dict to store the distribution(sigma and mean) of all params.
    Args:
        model: Network module of offspring.
        sigma_init: Initial value of sigma.
    Returns:
        mean_dict: Dict to store mean.
        sigma_dict: Dict to store sigam.
    Init:
        mean = L + (H-L) * rand
        sigma = 1 * sigma_init
    """
    sigma_dict, mean_dict = {},{}
    for n, p in model.named_parameters():
        sigma_dict[n] = torch.ones_like(p,dtype=torch.float) * sigma_init
        mean_dict[n] = torch.clamp(
            torch.ones_like(p,dtype=torch.float) * 1.0 + 
            torch.rand_like(p,dtype=torch.float) * (1.0 - 0.0),
            min = 0,
            max = 11
        )
    return mean_dict, sigma_dict

def sync_scalar(scalar, world_size):
    """sync scalar(value or vector) from all processes."""
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    if type(scalar) is int or float:
        scalar_one = torch.tensor([scalar]).to(device)
        scalar_all = [torch.tensor(scalar).clone().detach().requires_grad_(False).to(device) for _ in range(world_size)]
        #scalar_all = torch.zeros((world_size, 1), dtype=scalar_one.dtype)
    elif type(scalar) is list:
        scalar_one = torch.tensor(scalar,dtype=torch.float).to(device)
        scalar_all = [torch.tensor(scalar).clone().detach().requires_grad_(False) for _ in range(world_size)]
        # scalar_all = torch.zeros((world_size, scalar_one.shape[0]), dtype=scalar_one.dtype)
    dist.all_gather(scalar_all,scalar_one)
    return torch.tensor(scalar_all)

def sync_model(model, src):
    """Broadcast parameters from rank 0 to all other processes."""
    for param in model.parameters():
        dist.broadcast(param, src)
    
def cal_weight(sync_loss):
    """Calculate weight of each process."""
    sync_loss = sync_loss.detach().cpu().numpy()
    pop_size = len(sync_loss)

    rank = np.argsort(sync_loss) + 1
    tmp = [np.log(pop_size + 0.5) - np.log(r) for r in rank] # pop_size is lam in population (=len(rank?))
    w = tmp / np.sum(tmp)
    return w

def main(args):
    # set dist
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # set device
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # set model
    model = Model().to(device)
    if args.use_ddp:
        ddp_model = DDP(model,device_ids=[rank]).to(device)
    log(f"model")

    # set loss
    label = torch.Tensor([1,2]).to(device)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    if args.use_grad: 
        # set grad
        for p in model.parameters():
            p.requires_grad = True
            print(p.grad)  
        
        # configure optimizer
        adam = optim.Adam([p for p in model.parameters()], lr=0.1)
        adam.zero_grad()

        if args.use_ddp:               
            # forward
            out = ddp_model(torch.Tensor([1,2]))
        else:
            # forward
            out = model(torch.Tensor([1,2]))

        loss = loss_fn(label, out)
        loss.backward()

        log_params(adam)
        adam.step()
        log_params(adam)
    else:
        for p in model.parameters():
            p.requires_grad = False

        # configure optimizer of layer 0
        lr = 0.1
        if args.use_ddp:
            evo = EA([p for p in ddp_model.module.parameters()], lr=lr)
        else:
            evo = EA([p for p in model.parameters()],lr=lr)
        print(evo.param_groups[0]['params'][0])
        log(f"evo")
        # evo.set_grad(False)
        print(evo.param_groups) 

        # initial
        seed = rank
        torch.manual_seed(seed)    

        # build gauss dist for each param        
        mean, sigma = build_gauss(model, args.sigma_init) # dict, size = param_size

        if args.use_ddp:
            child = mutate(ddp_model, mean, sigma)
            log_params(child)

            ddp_model.set_params(child)
            output = ddp_model(torch.Tensor([1,2]))
        else:
            # mutate
            for n,p in model.named_parameters():
                epsilon = torch.randn_like(p, dtype=torch.float) * sigma[n] + mean[n]
                epsilon.to(device)

                p = torch.clamp(torch.add(p, epsilon), 0, 1) # clip
                log(f"Perturb parameter: {n} {p}")
    
            # inference
            input = torch.Tensor([1,2]).to(device)
            output = model(input)

        loss = loss_fn(label, output) #scalar
        # loss.backward()

        # sync loss and seed
        sync_loss = sync_scalar(loss, world_size)
        sync_seed = sync_scalar(seed, world_size)
        log(f"sync_loss: {sync_loss}")
        log(f"sync_seed: {sync_seed}")

        if args.use_select:
            selected = int(torch.argmax(sync_loss)) # select a node
            sync_model(model, selected) # sync model from selected node
            log(f"selected: {selected}")
        else:
            w = cal_weight(sync_loss) # average weight
            log(f"weight: {w}")
            
            log(f"Perturb parameter: {model.state_dict()}")
        
            for i in range(len(sync_loss)):
                for n, p in model.named_parameters():
                    torch.manual_seed(sync_seed[i])
                    epsilon = torch.randn_like(p, dtype=torch.float) * sigma[n] + mean[n]
                    epsilon.to(device)

                    # step
                    p = torch.add(p, lr * w[i] * epsilon)
                    p = torch.clamp(p, 0, 1) # clip
                    
            log(f"Perturb parameter: {model.state_dict()}")
                    
        # update
        #log_params(evo)
        #evo.step()
        #log_params(evo)            

def run(local_world_size, local_rank, args, func):
    def setup(rank, world_size):
        #os.environ['MASTER_ADDR'] = address
        #os.environ['MASTER_PORT'] = port
        dist.init_process_group(
            backend="nccl", 
            init_method="file:///home/yangqi/sharefile",
            world_size=world_size, 
            rank=rank
        )

    def cleanup():
        dist.destroy_process_group()    
    
    setup(local_rank, local_world_size)
    log(f"rank:{local_rank}")
    log(f"world_size:{local_world_size}")
    func(args)
    cleanup()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument("--use_grad", type=int, default=0)
    parser.add_argument("--use_ddp", type=int, default=0)
    parser.add_argument("--use_select", type=int, default=0)
    parser.add_argument("--sigma_init", type=int, default=0)
    args = parser.parse_args()
    return args

def test(args):
    import time
    time.sleep(5)
    print("test test test")

if __name__ == "__main__":
    # if grad use Adam else use EA
    args = parse()
    run(args.local_world_size, args.local_rank, args, main)
    #run(args.local_world_size, args.local_rank, args, main) 