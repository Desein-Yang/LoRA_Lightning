export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=1

python /home/yangqi/anaconda3/envs/NLU/lib/python3.7/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=2 test_eaopt.py --local_world_size=2 --use_ddp=0 --sigma_init=1 > 0426-1.txt
