#!/bin/bash
#python -V
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="89.72.32." --master_port="8848" trainer.py --model_id=0
#yhrun -n 1 -p gpu_v100_test sleep 1d
yhalloc -N 1 -p gpu_v100
#yhalloc -N 1 -p gpu_v100_test
