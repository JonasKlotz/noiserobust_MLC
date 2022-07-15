#!/bin/bash

#SBATCH -J ASL_TRAINING	# Job Name

#SBATCH --nodes=2               # Anzahl Knoten N
#SBATCH --ntasks-per-node=5     # Prozesse n pro Knoten
#SBATCH --ntasks-per-core=5	  # Prozesse n pro CPU-Core
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=60:00:00 # Erwartete Laufzeit

## AUf GPU Rechnen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node



#SBATCH -o logs/logfile_resnet_base                 # send stdout to outfile
#SBATCH -e logs/errfile_resnet_base                 # send stderr to errfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rs_3.8


echo Start

# Parameters for running
model=("resnet_base")
loss=("weighted_bce" "bce" "asl")
optim=("adam"  "sgd")
learning_rates=(0.001  0.005)



for m in ${model[@]}; do
	for l in ${loss[@]}; do
		for o in ${optim[@]}; do
				for lr in ${learning_rates[@]}; do
						args="-model ${m} -loss ${l} -optim ${o} -d_model ${d} -lr ${lr}"
						echo args
						python3 src/main.py $args

			done
		done
	done
done


