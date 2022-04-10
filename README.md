# Light Code for Parameter-efficient Finetune

## Usage

```bash
bash run_lora.sh
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
- v1.3 支持 Lora 
  - 新增 Lora Linear 和 Embedding 层
  - 新增 add/sum_trainable params()
  - 新增 optimizer 输出 log (优化 delta 分布)
