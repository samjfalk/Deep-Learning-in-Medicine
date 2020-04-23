#!/bin/bash
#

#SBATCH --job-name=run_py_script
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=slurm_%j.out

module purge
source activate /home/sjf374/dl4med
## any other module load or environment-related commands go here

srun python test2_conmat.py