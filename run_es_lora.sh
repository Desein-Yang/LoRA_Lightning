export num_gpus=1
export CUDA_VISIBLE_DEVICES=5
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./logs/GLUE/SST-2"
run_id="lora_ber_v1.4.0_test"
model="lora-ft"
for i in 1
do
    #python run_lora.py --run_id ${run_id} --task sst2 --gpus 1 --epoch 10 --batch 16 --lr 0.0005 --lora True --alpha 8 --r 8 > ./lora-ft/${run_id}-output${i}.log 
    python run_lora.py > ./logs/${model}/${run_id}-output${i}.log
done