#!/bin/sh
#SBATCH --job-name=coco_mobilenext
#SBATCH --account=project_2000255
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4,nvme:200
##SBATCH --mem-per-cpu=8000


srun --ntasks=1 --ntasks-per-node=1 \
    tar xf /scratch/project_2001284/anwer/convnext/mobilevit/dataset/coco/coco.tar.gz -C "$LOCAL_SCRATCH" --overwrite

timeout 35.5h python main_train.py --common.config-file config/detection/ssd_mobilenext_320.yaml --common.results-loc results_coco_mobilenext --dataset.root-train "$LOCAL_SCRATCH" --dataset.root-val "$LOCAL_SCRATCH"
if [[ $? -eq 124 ]]; then
  sbatch ./run_on_mahti.sh
fi