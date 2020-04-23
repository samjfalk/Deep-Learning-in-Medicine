#!/bin/bash
#

#SBATCH --job-name=rerun_w_more_epoch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=slurm_%j.out

module purge
source activate /home/sjf374/dl4med
## any other module load or environment-related commands go here

srun python test1.py