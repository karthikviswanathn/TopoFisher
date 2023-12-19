#!/bin/bash

#SBATCH -n 128  
#SBATCH -t 01:00:00
#SBATCH --job-name=vectorizePD
#SBATCH --output=/projects/0/gusr0688/vk/cmd_outputs/pi_imnn_R_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.viswanathan@uva.nl

input_file_loc="$1"
output_file_loc="$2"
num_iterations="$3"
machine_type="$4"
if [ "$machine_type" == "local" ]; then
    parent_dir="/Users/karthikviswanathan/Desktop/TDA/codes/fisherPH/versions/TopoFisher/topofisher"
else
    parent_dir="/projects/0/gusr0688/vk/TopoFisher/topofisher"
    source activate newEnvbkUp
    echo "Loaded virtual environment."
fi
imnn_script_loc="$parent_dir"/cluster/codes/imnn.py

python "$imnn_script_loc" "$input_file_loc" "$output_file_loc" "$num_iterations"