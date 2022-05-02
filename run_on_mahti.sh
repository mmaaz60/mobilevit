#!/bin/sh
#SBATCH --job-name=ConvNeXt
#SBATCH --account=project_2000255
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4,nvme:400
##SBATCH --mem-per-cpu=8000


srun --ntasks=2 --ntasks-per-node=1 \
    tar xf /scratch/project_2001284/anwer/mvit_pretraining/data/train.tar.gz -C "$LOCAL_SCRATCH" --overwrite

srun --ntasks=2 --ntasks-per-node=1 \
    tar xf /scratch/project_2001284/anwer/mvit_pretraining/data/val.tar.gz -C "$LOCAL_SCRATCH" --overwrite

PYTHONWARNINGS="ignore::Warning" timeout 35.5h srun python main_train.py --common.config-file config/classification/convnext.yaml --common.results-loc results/convnext --dataset.root_train "$LOCAL_SCRATCH/train" --dataset.root_val "$LOCAL_SCRATCH/val" --dataset.workers 10
if [[ $? -eq 124 ]]; then
  sbatch ./run_on_mahti.sh
fi