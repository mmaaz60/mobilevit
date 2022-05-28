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
    tar xf /scratch/project_2001284/anwer/convnext/mobilevit/dataset/pascal_coco/pascal_coco.tar.gz -C "$LOCAL_SCRATCH" --overwrite

timeout 35.5h cvnets-train --common.config-file config/segmentation/deeplabv3_mobilenext.yaml --common.results-loc results_deeplabv3_mobilenext --dataset.root-train "$LOCAL_SCRATCH/pascal_voc/VOCdevkit" --dataset.root-val "$LOCAL_SCRATCH/pascal_voc/VOCdevkit" --dataset.pascal.coco-root-dir "$LOCAL_SCRATCH/coco_preprocess"
if [[ $? -eq 124 ]]; then
  sbatch ./run_on_mahti.sh
fi