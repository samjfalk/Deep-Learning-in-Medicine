#!/bin/bash
#

#SBATCH --job-name=resnet18
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:1
#SBATCH --time=5:00:00
#SBATCH --output=slurm_%j.out

module purge
source activate /home/sjf374/dl4med
## any other module load or environment-related commands go here

srun python resnet18.py