export num_gpus=1
export CUDA_VISIBLE_DEVICES=7
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./logs/GLUE/SST-2"
#run_id="lora_bert_0413_lr_5e-4_batch-32_a-8_r-8"
#run_id="roberta_warmup_test"
run_id="bert_standard"
model="bert-ft"
for i in 1
do
    #python run_lora.py --run_id ${run_id} --task sst2 --gpus 1 --epoch 10 --batch 16 --lr 0.0005 --lora True --alpha 8 --r 8 > ./lora-ft/${run_id}-output${i}.log 
    python run_lora.py > ./logs/${model}/${run_id}-output${i}.log
done