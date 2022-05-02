#!/bin/sh
#SBATCH --job-name=convnext
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:16
##SBATCH --gpus=16


srun python main_train.py --common.config-file config/classification/convnext.yaml --common.results-loc results/convnext --dataset.root-train /nfs/projects/mbzuai/salman/imagenet_1k/train --dataset.root-val /nfs/projects/mbzuai/salman/imagenet_1k/val --dataset.workers 6 --dataset.train-batch-size0 64