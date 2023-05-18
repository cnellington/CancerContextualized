#!/bin/bash

#SBATCH --job-name=cancontext
# partition (queue) declaration
#SBATCH -t 3-00:00
#SBATCH --partition=dept_cpu
#SBATCH --mail-user=xiaoh@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=16
#SBATCH --clusters=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

module load cuda/11.7.1

python disease_cv.py 
