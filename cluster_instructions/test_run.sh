#!/bin/bash

#SBATCH -J test	# Job Name

#SBATCH --nodes=1               # Anzahl Knoten N
#SBATCH --ntasks-per-node=5     # Prozesse n pro Knoten
#SBATCH --ntasks-per-core=5	  # Prozesse n pro CPU-Core
#SBATCH --mem=10G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=00:05:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=standard
##SBATCH --gres=gpu:tesla:1                      # Use 1 GPU per node


#SBATCH -o logs/logfile                  # send stdout to outfile
#SBATCH -e logs/errfile                  # send stderr to errfile

module load python/3.7.1 
module load comp/gcc/7.2.0
module load nvidia/cuda/11.2
module load nvidia/cudnn/9.0 


##cd
source /home/users/j/jonasklotz/remotesensing/rs_env/bin/activate

echo Start

##chmod +x ~workingModel.py

python3 cuda_test.py

