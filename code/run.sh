#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=rico_train
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 48:00:00

python -m torch.distributed.launch --use-env --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/RICO_scannet.conf --scan_id 1 --nepoch 5 --infer_stage 1