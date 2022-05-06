#!/bin/sh
#SBATCH --job-name=coco_mobilenext
#SBATCH --account=project_2000255
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4,nvme:200
##SBATCH --mem-per-cpu=8000


timeout 35.5h cvnets-train --common.config-file config/detection/ssd_mobilenext_320.yaml --common.results-loc results_coco_mobilenext
if [[ $? -eq 124 ]]; then
  sbatch ./run_on_mahti.sh
fi