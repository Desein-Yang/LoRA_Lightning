import torch
import torch.distributed as dist
import os
from typing import List, Optional
from torch import Tensor

def setup(rank: int, world_size: int, init_methods= Optional[str]):
    """Set up torch distributed with {word_size} nodes."""
    #os.environ['MASTER_ADDR'] = address
    #os.environ['MASTER_PORT'] = port
    if init_methods is None:
        init_methods = "/home/yangqi/sharefile"

    #if os.path.isfile(init_methods): # gurantee it don't exist
    #    os.remove(init_methods)

    dist.init_process_group(
        backend="nccl", 
        init_method="file://" + init_methods,
        world_size=world_size, 
        rank=rank
    )
    print(f"Start Distributed Training with {world_size} nodes.")
    print(f"rank        :{rank}")
    print(f"world_size  :{world_size}")

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

def flatten_tensor_dict(tensor_list:dict):
    return torch.cat([p.flatten() for p in tensor_list.values()])