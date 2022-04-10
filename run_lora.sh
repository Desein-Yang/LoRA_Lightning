export num_gpus=1
export CUDA_VISIBLE_DEVICES=6
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./logs/GLUE/SST-2"

for i in 1
do
    python run_lora.py --run_id "0408_lr_5e-4_batch-32_a-8_r-8" --task sst2 --gpus 1 --epoch 10 --batch 32 --lr 0.00005 --lora True --alpha 8 --r 8 > output${i}.log 
done


