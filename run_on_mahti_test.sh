#!/bin/sh
#SBATCH --job-name=ConvNeXt_Test
#SBATCH --account=project_2000255
#SBATCH --partition=gputest
#SBATCH --time=00:06:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4,nvme:500
##SBATCH --mem-per-cpu=8000


srun --ntasks=1 --ntasks-per-node=1 \
    tar xf /scratch/project_2001284/anwer/mvit_pretraining/data/tiny_imagenet/train.tar.gz -C "$LOCAL_SCRATCH" --overwrite

srun --ntasks=1 --ntasks-per-node=1 \
    tar xf /scratch/project_2001284/anwer/mvit_pretraining/data/tiny_imagenet/val.tar.gz -C "$LOCAL_SCRATCH" --overwrite

PYTHONWARNINGS="ignore::Warning" timeout 4m srun python main_train.py --common.config-file config/classification/convnext.yaml --common.results-loc results/convnext --dataset.root_train "$LOCAL_SCRATCH/train" --dataset.root_val "$LOCAL_SCRATCH/val" --dataset.workers 10
if [[ $? -eq 124 ]]; then
  sbatch ./run_on_mahti_test.sh
fi