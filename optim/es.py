from distutils.log import info
import time
import logging
import torch
import numpy as np
import torch.distributed as dist

from torch import Tensor, tensor
from typing import List, Optional
from torch.optim import Optimizer
from .utils import flatten_tensor_list

# get 
# optimizer = torch.optim.Adam(bert.parameters(),lr=0.01)

# source code
# Optimizer : https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
# Adam : https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py


# Example
# Import 
# from optim.evolution import EA
# from optim.utils import sync_params, sync_scalar
# 
# Build optim
# evo = EA([p for p in model.parameters()], lr=lr, select=use_select)
# 
# Optimize 
# for step in range(max_step):
#    evo.mutate()
#
#    loss = model(input)
#    
#    sync_loss = sync_scalar(loss, world_size)
#    sync_seed = sync_scalar(seed, world_size)
#    evo.step(sync_loss, sync_seed)
class EvoStrategy(Optimizer):
    r"""Implements Evolution Strategy Algorithm
    
    .. math::
        \begin{aligned}
        \end{aligned}
    
    For further details regarding the algorithm we refer to: Wierstra, Daan, et al. "Natural evolution strategies." The Journal of Machine Learning Research 15.1 (2014): 949-980.

    Args:
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    foreach (bool, optional): whether foreach implementation of optimizer is used (default: None)
    maximize (bool, optional): maximize the params based on the objective, instead of minimizing (default: False)
    verbose (bool, optional): whether to print the progress (default: True)

    Examples:
    #Import 
    from optim.evolution import EA
    from optim.utils import sync_params, sync_scalar

    #Build optim
    evo = EA([p for p in model.parameters()], lr=lr, select=use_select)

    #Optimize 
    for step in range(max_step):
        evo.mutate()

        loss = model(input)
        
        sync_loss = sync_scalar(loss, world_size)
        sync_seed = sync_scalar(seed, world_size)
        evo.step(sync_loss, sync_seed)

    """
    def __init__(self, params, 
        lr=1e-3, eps=1e-8,
        sigma:float = 1,
        clip: tuple = (-1,1),
        select: bool = True, # use select machnism 
        maximize: bool = False, 
        foreach: Optional[bool] = None,
        verbose: Optional[bool] = True
        ) -> None:
        defaults = dict(lr=lr, eps=eps, sigma=sigma,  clip=clip, maximize=maximize, foreach=foreach)
        super().__init__(params, defaults)
        
        self.mean, self.sigma = self.build_gauss(sigma_init=sigma, low=clip[0], high=clip[1])
        
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.device = torch.device("cuda", self.rank)
        self.select= select
       
        assert clip[1] > clip[0], "clip[1] should be larger than clip[0]"
        self.clip = clip

        self.set_grad(False)
        self.log_params("Init")

    def __setstate__(self, state: dict) -> None:
        return super().__setstate__(state)

    def params_no_grad(self):
        params_no_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad is False:
                    params_no_grad.append(p)
        return params_no_grad

    @torch.no_grad()
    def step(self, closure=None, **kwargs) -> Optional[float]:
        sync_loss = kwargs['loss']
        sync_seed = kwargs['seed']
        
        # get the current params for optimize    
        params_no_grad = self.params_no_grad()
        for n, p in enumerate(params_no_grad):
            state = self.state[str(n)]
            if len(state) == 0:
                state['step'] = torch.tensor(0.)
                state['weight'] = torch.zeros(self.size,dtype=torch.float)
                state['seed'] = torch.zeros(self.size,dtype=torch.float)

        if self.select:
            loss = self.step_best(params_no_grad, sync_loss, sync_seed, closure)
        else:
            loss = self.step_accu(params_no_grad, sync_loss, sync_seed, closure)
        return loss
    
    @torch.no_grad()
    def step_accu(self, params, sync_loss, sync_seed, closure=None):
        # calculate weight list
        weight = self._cal_weight(sync_loss)

        tem_pl = []
        reconstruct_el = []
        for i in range(len(sync_seed)): # avoid switch the seed too frequently
            torch.manual_seed(sync_seed[i])
            for param_id, p in enumerate(params):
                # update state
                state = self.state[str(param_id)]
                state['step'] += 1
                state['weight'] = weight
                state['seed'] = sync_seed
                
                sigma = self.sigma[param_id]
                mean = self.mean[param_id]

                # test
                sigma, mean = 1,0

                epsilon = torch.randn_like(p, dtype=torch.float) * sigma + mean
                epsilon.to(self.device)
                    
                reconstruct_el.append(epsilon)

                if self.rank == i: # decrease noise to recover base model
                    p.add_(epsilon * -1)
                    tem_pl.append(p)

                p.add_(epsilon * weight[i])
                # p = torch.clamp(torch.add(p,epsilon), min=self.clip[0], max=self.clip[1])
        
        self.log(f"Reconstruct Epsilon: {flatten_tensor_list(reconstruct_el)}")
        self.log(f"Temporart Model    : {flatten_tensor_list(tem_pl)}")

        # OpenAI Es do not need to update sigma and mean
        # update mean
        # self.mean[num] = param.clone()
        # update sigma
        # self.sigma[num] = sigma * self._cal_lambda
        return sync_loss

    @torch.no_grad()
    def step_best(self, params, sync_loss, sync_seed=None, closure=None):
         # modify state e.g., step
        for param_id, p in enumerate(params):
            state = self.state[str(param_id)]
            state['steps'] += 1
    
            # update params
            selected = int(torch.argmax(sync_loss))# select a node
            self._sync_param(p, selected) # sync model from selected node
        
        return sync_loss

    # ! deprecated
    # ! becasuse the params_id could not gurantee the order
    # ! but the param in optim do not have name to construct a dict
    def build_gauss(self, sigma_init:float, low:float, high:float,):
        """Build a list to store the distribution(sigma and mean) of all params.
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
        params_no_grad = self.params_no_grad()
        sigma_list, mean_list = [],[]
        for n,p in enumerate(params_no_grad):
            sigma_list.append(torch.ones_like(p, dtype=torch.float) * sigma_init)
            # uniform distribution in [low, high]
            mean = torch.ones_like(p, dtype=torch.float) * low + torch.rand_like(p, dtype=torch.float) * (high - low)
            mean = torch.clamp(mean, min=low, max=high)
            mean_list.append(mean)
             
        return mean_list, sigma_list

    def mutate(self): 
        # use current time n second as seed
        torch.manual_seed(int(time.time_ns())+self.rank)    
        self.log(f"Mutate Seed: {torch.initial_seed()}")
        epsilon_l = []
        params_no_grad = self.params_no_grad()
        for param_id,p in enumerate(params_no_grad):
            sigma = self.sigma[param_id]
            mean = self.mean[param_id]

            # test
            sigma,mean = 1,0
            epsilon = torch.randn_like(p, dtype=torch.float) * sigma + mean
            epsilon.to(self.device)
                
            p.add_(epsilon)
            # p = torch.clamp(p, min=self.clip[0], max=self.clip[1])
            epsilon_l.append(epsilon)
        

        self.log(f"Epsilon: {flatten_tensor_list(epsilon_l)}")
        #self.log_params("Mutated")
        
    def set_grad(self, require_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad = require_grad

    def log_state(self, step):
        self.log(f"========= Step:{step} =========")
        for key in self.state.key():
            self.log(self.state[key])
        self.log("==================================")

    # only use for small model
    def log_params(self, msg):
        flatten_params = flatten_tensor_list(self.param_groups[0]['params'])
        self.log(f" {msg} Params: {flatten_params}")

    def log(self, msg):
        rank = self.rank
        logging,info(f"[{rank}] {msg}")

    @staticmethod
    def _cal_weight(sync_loss : torch.Tensor):
        """Calculate weight of each process."""
        sync_loss = sync_loss.detach().cpu().numpy()
        pop_size = len(sync_loss)

        rank = np.argsort(sync_loss) + 1
        tmp = [np.log(pop_size + 0.5) - np.log(r) for r in rank] # pop_size is lam in population (=len(rank?))
        w = tmp / np.sum(tmp)
        return w        

    @staticmethod
    def _cal_lambda(steps_passed: int, steps_max: int):
        """ 计算 lambda，用于计算 sigma
        返回值：
            lambda : float"""
        return torch.rand() * (0.1-0.1*steps_passed/steps_max) + 1.0


# ! deprecated
def ea(self,
    params: List[Tensor],
    state_steps : List[Tensor],
    delta : List[Tensor],
    lr : float,
    eps: float,
    maximize : bool,
    foreach :bool,
):
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
        state_steps,
        delta = delta,
        lr = lr,
        eps = eps,
        maximize=maximize,
        foreach = foreach,
    )    

def _single_tensor_ea(
    params: List[Tensor],
    delta : List[Tensor],
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
        
        # update (addcdiv(input,t1,t2,*,value=1) = =input + valuext1/t2)
        param.addcdiv_(tensor1=delta, tensor2=param, value=-step_size) 

def _multi_tensor_ea(
    params: List[Tensor],
    state_steps : List[Tensor],
    delta : List[Tensor],
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
    torch._foreach_addcdiv(params, delta, step_size)



