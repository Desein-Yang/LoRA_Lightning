export num_gpus=1
export CUDA_VISIBLE_DEVICES=3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./logs/GLUE/SST-2"
run_id="20220408"

for i in 1
do
    #python run_bert_finetune.py --run_id "20220329" --task sst2 --gpus 1 --epoch 1 --batch 32 --lr 0.00002 --warmup_ratio 1e5 > analysis${i}.log 
    python run_bert_finetune.py --run_id ${run_id} --task sst2 --gpus 1 --epoch 10 --batch 32 --lr 0.00002 --warmup_ratio 0.06 > ./roberta-ft/${run_id}/output${i}.log 
done


