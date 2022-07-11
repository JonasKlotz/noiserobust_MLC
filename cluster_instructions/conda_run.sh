#!/bin/bash

#SBATCH -J ASL_TRAINING	# Job Name

#SBATCH --nodes=1               # Anzahl Knoten N
#SBATCH --ntasks-per-node=5     # Prozesse n pro Knoten
#SBATCH --ntasks-per-core=5	  # Prozesse n pro CPU-Core
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=80:00:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node


#SBATCH -o logfile                  # send stdout to outfile
#SBATCH -e errfile                  # send stderr to errfile




##cd
##source /home/users/j/jonasklotz/rs_env/bin/activate


echo Start

python3 /home/users/j/jonasklotz/remotesensing/src/main.py -model resnet_base -dataset deepglobe -d_model 50 -epoch 50 -lr 0.0001 -loss asl -optim adam
python3 /home/users/j/jonasklotz/remotesensing/src/main.py -model lamp -dataset deepglobe -d_model 50 -epoch 50 -lr 0.0001 -loss asl -optim adam
python3 /home/users/j/jonasklotz/remotesensing/src/main.py -model CbMLC -dataset deepglobe -d_model 50 -epoch 50 -lr 0.0001 -loss asl -optim adam


