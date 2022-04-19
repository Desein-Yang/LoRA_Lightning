from email.policy import default
import torch
from torch import Tensor
from typing import List, Optional
from torch.optim import Optimizer
import numpy as np

# get 
# optimizer = torch.optim.Adam(bert.parameters(),lr=0.01)

# source code
# Optimizer : https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
# Adam : https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py


class EAOptimizer(Optimizer):
    r"""Implements Algorithm
    
    Args:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    foreach (bool, optional): whether foreach implementation of optimizer
        is used (default: None)
    maximize (bool, optional): maximize the params based on the objective, instead of
        minimizing (default: False)
    """
    def __init__(self, params, lr=1e-3, eps=1e-8, 
        maximize: bool = False, 
        foreach: Optional[bool] = None) -> None:
        
        defaults = dict(lr=lr, eps=eps, maximize=maximize, foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        return super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            exp_avgs, exp_avg_sqs, max_exp_avg_sqs = [],[],[]
            state_steps = []

            for p in group['params']:
                params_with_grad.append(p)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avgs'] = torch.zeros_like(p,memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avgs'])
                exp_avg_sqs.append(state['exp_avg_sqs'])

                state_steps.append(state['step'])

        ea(params_with_grad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=group['lr'],
        eps=group['eps'],
        maximize=group['maximize'],
        foreach=group['foreach'],)
        return loss


def ea(
    params: List[Tensor],
    # grads: List[Tensor],
    exp_avgs : List[Tensor],
    exp_avg_sqs : List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps : List[Tensor],
    lr : float,
    eps: float,
    maximize : bool,
    foreach :bool,
    ):
    """Functional API"""
    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, state_steps argument must contain a list of single tensors")
  
    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_ea
    else:
        func = _single_tensor_ea

    func(params,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         lr = lr,
         eps = eps,
         maximize=maximize,
         foreach = foreach,
    )

def _single_tensor_ea(
    params: List[Tensor],
    # grads: List[Tensor],
    exp_avgs : List[Tensor],
    exp_avg_sqs : List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps : List[Tensor],
    lr : float, 
    eps : float,
    maximize : bool,
    foreach : bool,
):
    step_t = state_steps[i]
    step_t += 1
    step = step_t.item()

    for i, param in enumerate(params):
        # process
        delta = torch.ones_like(param)

        step_size = lr * delta 
        
        # update
        param.addcdiv_(delta, param, step_size) 

def _multi_tensor_ea(
    params: List[Tensor],
    # grads: List[Tensor],
    exp_avgs : List[Tensor],
    exp_avg_sqs : List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps : List[Tensor],
    lr : float,
    eps : float,
    maximize : bool,
    foreach : bool,
):
    if len(params) == 0:
        return 
    
    # update steps
    torch._foreach_add_(state_steps, 1)

    if maximize:
        delta = torch._foreach_neg(tuple(delta))

    step_size = lr * delta
    # update
    torch._foeach_addcdiv(params, delta, step_size)

def select(pop, fitness, pop_size):    # nature selection wrt pop's fitness
    idx = np.random.choice(
        np.arange(pop_size), 
        size=pop_size, replace=True,
        p=fitness/fitness.sum()
    )
    return pop[idx]
       
def mutate(child):
    pass

def get_fitness(pop):
    pass