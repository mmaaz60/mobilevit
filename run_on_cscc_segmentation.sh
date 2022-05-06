#!/bin/sh
#SBATCH --job-name=coco_mobilenext
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=64
#SBATCH --qos=gpu-8

LOCAL_SCRATCH='/home/muhammad.maaz/mobilevit/dataset'

cvnets-train --common.config-file config/segmentation/deeplabv3_mobilenext.yaml --common.results-loc results_deeplabv3_mobilenext --dataset.root-train "$LOCAL_SCRATCH/pascal_voc/VOCdevkit" --dataset.root-val "$LOCAL_SCRATCH/pascal_voc/VOCdevkit" --dataset.pascal.coco-root-dir "$LOCAL_SCRATCH/coco_preprocess"