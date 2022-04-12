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
- v1.3.0 支持 Lora 
  - 新增 Lora Linear 和 Embedding 层
  - 新增 add/sum_trainable params()
  - 新增 optimizer 输出 log (优化 delta 分布)
  - 新增 README.md
  - 调试收敛正常成功，复现结果 91.3%
  - v1.3.1 Fix bugs : 
    - log_opt params 无法正确调用所有 delta, 新增 flatten_params 方法 
    - warmup step 过大、计算优化步数多乘了 batchsize
    - 其他优化，shell 脚本调整，删掉多余空行
