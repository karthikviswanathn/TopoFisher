#!/bin/bash

#SBATCH -n 64  
#SBATCH -t 01:00:00

#SBATCH --job-name=vectorizePD
#SBATCH --output=/projects/0/gusr0688/vk/cmd_outputs/piFromPD_R_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.viswanathan@uva.nl


module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load SciPy-bundle/2021.05-foss-2021a

echo "Loaded libraries. Running the code now" 

# Check if the number of arguments is correct
if [ $# -ne 3 ]; then
    echo $#
    echo "Usage: $0 <start> <increment> <num_iterations>"
    exit 1
fi

# Parse command line arguments
fid_increment="$1"
der_increment="$2"
num_iterations="$3"

# Define an array to store the ranges
ranges=()
fid_start=0
der_start=0
# Use a for loop to generate the ranges and append them to the array
for ((i = 1; i <= num_iterations; i++)); do
    fid_end=$((fid_start + fid_increment))
    der_end=$((der_start + der_increment))	
    output="/projects/0/gusr0688/vk/outputs/bdp/topk_vecs${i}.pkl"
    ranges+=("$fid_start $fid_end $der_start $der_end $output")
    
    # Increment the start value
    fid_start=$fid_end
    der_start=$der_end
done

# Loop through the ranges and execute the commands
for range_args in "${ranges[@]}"; do
    echo "ranges = $range_args"
    python3 /projects/0/gusr0688/vk/topofisher/piFromPD.py $range_args &
done

# Wait for all background jobs to finish
wait






