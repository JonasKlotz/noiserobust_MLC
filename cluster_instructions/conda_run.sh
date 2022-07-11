#!/bin/bash

#SBATCH -J ASL_TRAINING	# Job Name

#SBATCH --nodes=1               # Anzahl Knoten N
#SBATCH --ntasks-per-node=5     # Prozesse n pro Knoten
#SBATCH --ntasks-per-core=5	  # Prozesse n pro CPU-Core
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=00:30:00 # Erwartete Laufzeit

## AUf GPU Rechnen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node



#SBATCH -o logs/logfile_conda                  # send stdout to outfile
#SBATCH -e logs/errfile_conda                  # send stderr to errfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rs_3.8


echo Start

python3 /src/main.py -model CbMLC -dataset deepglobe -d_model 50 -epoch 50 -lr 0.0001 -loss asl -optim adam


