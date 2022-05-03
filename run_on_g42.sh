#!/bin/sh
#SBATCH --job-name=convnext
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:16
##SBATCH --gpus=16


srun python main_train.py --common.config-file "$1" --common.results-loc "$2" --dataset.root-train /nfs/projects/mbzuai/salman/imagenet_1k/train --dataset.root-val /nfs/projects/mbzuai/salman/imagenet_1k/val --dataset.workers 6 --dataset.train-batch-size0 64

# Example run -> sbatch run_on_g42.sh <path to .yaml config> <path to the output dir>