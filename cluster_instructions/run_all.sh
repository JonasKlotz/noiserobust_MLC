#!/bin/bash

sbatch cluster_instructions/cbmlc_asl.sh
sbatch cluster_instructions/cbmlc_bce.sh
sbatch cluster_instructions/cbmlc_wbce.sh

sbatch cluster_instructions/lamp_asl.sh
sbatch cluster_instructions/lamp_bce.sh
sbatch cluster_instructions/lamp_wbce.sh

sbatch cluster_instructions/resnet_asl.sh
sbatch cluster_instructions/resnet_bce.sh
sbatch cluster_instructions/resnet_wbce.sh