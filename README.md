# EvoBert


This is an implemention of
- Bert Finetune
- Low-Rank Finetune(A Parameter-efficient  Methods refers to https://github.com/microsoft/LoRA)
- Low-Rank Evolution Finetune (A PyTorch Optimizer with Evolution Strategy)


## Usage

```bash
bash run_lora.sh
bash run_bert.sh
bash run_es_lora.sh
```

## version
- v1.0 基于 pytorch lightning 实现 Bert finetune 
  - 新增 tf 和 csv 双 Loggers
  - max seq length = 64
- v1.1 细节优化
  - 支持从 config.ini 配置传入参数
  - 新增学习率调整 lr scheduler
- v1.2 支持 Roberta 模型 finetune
  - 新增学习率预热，通过参数 warmup ratio 和 step 调
  - 支持每个 epoch 保存最优 Checkpoint 
- v1.3 支持 Lora 低秩矩阵
  - 新增 Lora Linear 和 Embedding 层
  - 新增 add/sum_trainable params()
  - 新增 optimizer 输出 log (优化 delta 分布)
  - 新增 README.md
  - 调试收敛正常成功，复现结果 91.3%
  - v 1.3.1 Fix bugs : 
    - log_opt params 无法正确调用所有 delta, 新增 flatten_params 方法 
    - warmup lr scheduler 计算出错，没有 step()，修改后复现结果 92.55%
    - 其他优化，shell 脚本调整，删掉多余空行
  - v 1.3.2 Merge Code and Keep Clean:
    - 对照差异合并 Base 和 Lora 代码，统一 config 导入、model_id 识别和 warmup 机制
    - 归纳为 finetuner module 导入
    - 新增可控参数 max_seq_len, early_stop, model_check, model_id 等
    - 解决遗留不够整洁的代码目录
  - v 1.3.3 Merge code and Keep Clean:
    - 解决 lr scheduler 提前终止的 bug（多加了一个step）
    - 在 lora finetuner 中统一支持不同模型和算法
    - 确定了超参数设置
- v1.4 支持自定义 ea optim 
    - v1.4.0 支持自定义 optim 类(简单累加delta)
      - 测试了 torch ddp
      - 测试了 optimizer + torch dist 优化步骤正确
      - 测试了 optimizer + lora finetune 


## File Tree
├── functions
├── src
│  ├── articles
│  ├── components
│  │  ├── builder
│  │  │  ├── center
│  │  │  ├── left
│  │  │  │  └── sections
│  │  │  ├── lists
│  │  │  └── right
│  │  │    └── sections
│  │  ├── dashboard
│  │  ├── landing
│  │  ├── router
│  │  └── shared
│  └── constants
└── static
  └── images
    ├── screenshots
    └── templates

# EA Optimizer Example
```python
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
```
