#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/scratch/batch_jobs/code-prot/output-%j.out
#SBATCH --error=/users/rvinod/scratch/batch_jobs/code-prot/output-%j.err


#SBATCH --cpus-per-task=4

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

 
# Request an hour of runtime:
#SBATCH --time=72:00:00
#SBATCH --mem=60G

#SBATCH -J joint_geom


# Run a command
source ~/scratch/venvs/e3_diff/bin/activate

module load python/3.7.4


python3 data/joint_dataloader.py 
