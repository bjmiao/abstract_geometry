#!/bin/bash
#SBATCH --job-name=generalize
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem 4G
#SBATCH --output=log/hypercube_%j.out
#SBATCH --error=log/%j.err
#SBATCH --qos=urgent

eval "$(conda shell.bash hook)" # this is needed for conda activate (but why?)
conda activate py39-base
python $1 $2 $3 $4
