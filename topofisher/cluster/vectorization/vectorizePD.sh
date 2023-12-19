#!/bin/bash

#SBATCH -n 128  
#SBATCH -t 01:00:00

#SBATCH --job-name=vectorizePD
#SBATCH --output=/projects/0/gusr0688/vk/cmd_outputs/pi_imnn_R_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vkarthik095@gmail.com

input_file_loc="$1"
output_file_loc="$2"
vec_type="$3"
fid_start="$4"
der_start="$5"
num_iterations="$6"
k="$7"
machine_type="$8"
if [ "$machine_type" == "local" ]; then
    parent_dir="/Users/karthikviswanathan/Desktop/TDA/codes/fisherPH/versions/TopoFisher/topofisher"
else
    parent_dir="/projects/0/gusr0688/vk/TopoFisher/topofisher"
    source activate newEnvbkUp
    echo "Loaded virtual environment."
fi

json_file_loc="$parent_dir"/cluster/vectorization/jsons/"$vec_type".json

# Parse command line arguments
fid_increment=20
der_increment=1

# Define an array to store the ranges
ranges=()
# Use a for loop to generate the ranges and append them to the array
# num_iterations is the num_jobs
for ((i = 1; i <= num_iterations; i++)); do
    fid_end=$((fid_start + fid_increment))
    der_end=$((der_start + der_increment))	
    ranges+=("$fid_start $fid_end $der_start $der_end $k")
    
    # Increment the start value
    fid_start=$fid_end
    der_start=$der_end
done
for range_args in "${ranges[@]}"; do
    echo "ranges = $range_args"
    python3 "$parent_dir"/cluster/vectorization/vectorizePD.py "$input_file_loc" "$output_file_loc"\
     "$json_file_loc" $range_args &
done
wait
