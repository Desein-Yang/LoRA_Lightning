export num_gpus=1
export CUDA_VISIBLE_DEVICES=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./logs/GLUE/SST-2"

for i in 1 2 3
do
    python run_bert_finetune.py --run_id "20220329" --task sst2 --gpus 1 --epoch 3 --batch 16 --lr 0.00005 > out${i}.log 
done


