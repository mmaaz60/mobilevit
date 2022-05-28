#!/bin/sh
#SBATCH --job-name=convnext
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 #16
##SBATCH --gpus=1 #16


srun python main_train.py  --common.stats_only True --common.config-file "/nfs/projects/mbzuai/ashaker_2/mobilevit/config/detection/ssd_mobilenext_320.yaml" --common.results-loc /result_detection2 --dataset.root-train /nfs/projects/mbzuai/salman/imagenet_1k/train --dataset.root-val /nfs/projects/mbzuai/salman/imagenet_1k/val --dataset.workers 8 --dataset.train-batch-size0 64

# Example run -> sbatch run_on_g42.sh <path to .yaml config> <path to the output dir>
