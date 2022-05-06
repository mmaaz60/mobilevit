#!/bin/sh
#SBATCH --job-name=mobilenext_test
#SBATCH --account=project_2000255
#SBATCH --partition=gputest
#SBATCH --time=00:06:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4,nvme:100
##SBATCH --mem-per-cpu=8000


timeout 4m cvnets-train --common.config-file config/detection/ssd_mobilenext_320.yaml --common.results-loc results_coco_mobilenext_test
if [[ $? -eq 124 ]]; then
  sbatch ./run_on_mahti_test.sh
fi