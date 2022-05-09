import torch
from torch import Tensor
from typing import List
import pynvml

def record_time(func,*args,**kwargs):
    # warm up
    with torch.no_grad():
        for i in range(5):
            func(*args,**kwargs)
    
    # synchronize cpu task ends
    torch.cuda.synchronize()

    # record time cuda Event
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    starter.record()
    func(*args,**kwargs)
    ender.record()

    torch.cuda.synchronize()
    curr = starter.elapsed_time(ender)

    func_name = func.__name__
    print(str(func_name) + ':' + str(curr))
    return curr

def warmup_fn(warmup_steps, total_steps, type="linear"):
    def linear(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        ) 

    def poly(current_step):
        if current_step < warmup_steps:
            return (current_step / warmup_steps) ** 2
        return max(
            0.0, (float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))) ** 2
        )

    if type == "linear":
        lr_lambda = linear
    else:
        lr_lambda =  poly

    return lr_lambda

def flatten_tensor_list(tensor_list:List[Tensor]):
    return torch.cat([p.flatten() for p in tensor_list])

def flatten_tensor_dict(tensor_list:dict):
    return torch.cat([p.flatten() for p in tensor_list.values()])       

def sum_grad_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sum_trainable_params(trainable_params):
    return sum(p.numel() for p in trainable_params)

def idle_device(size):
    pynvml.nvmlInit()
    gpu_nums = pynvml.nvmlDeviceGetCount()
    gpu_free = []
    for i in range(gpu_nums):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_free.append(meminfo.free)
    idle_device = [gpu_free.index(i)-1 for i in sorted(gpu_free)[-1*size:]]
    pynvml.nvmlShutdown()
    return idle_device