import torch
import torch.distributed as dist
import os
from typing import List 
from torch import Tensor, Optional

def setup(rank: int, world_size: int, init_methods= Optional[str]):
    """Set up torch distributed with {word_size} nodes."""
    #os.environ['MASTER_ADDR'] = address
    #os.environ['MASTER_PORT'] = port
    if init_methods is None:
        init_methods = os.path.join('~','sharefile') # "file:///home/yangqi/sharefile"
    dist.init_process_group(
        backend="nccl", 
        init_method=init_methods,
        world_size=world_size, 
        rank=rank
    )

def cleanup():
    """Clean up torch distributed."""
    dist.destroy_process_group()    

def sync_scalar(scalar, world_size: int):
    """Sync scalar(value or vector) from all processes."""
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

def sync_params(params, src):
    """Broadcast parameters from rank 0 to all other processes."""
    for p in params:
        dist.broadcast(p, src)
    return params
    
def flatten_tensor_list(tensor_list:List[Tensor]):
    return torch.cat([p.flatten() for p in tensor_list])
