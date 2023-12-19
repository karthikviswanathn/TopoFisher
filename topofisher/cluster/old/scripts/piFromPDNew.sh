#!/bin/bash

#SBATCH -n 2048  
#SBATCH -t 03:00:00

#SBATCH --job-name=vectorizePD
#SBATCH --output=/projects/0/gusr0688/vk/cmd_outputs/piFromPD.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.viswanathan@uva.nl

module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load SciPy-bundle/2021.05-foss-2021a

cp -r /projects/0/gusr0688/vk/pd_sancho "$TMPDIR"

echo "Loaded libraries. Running the code now" 

# Check if the number of arguments is correct
if [ $# -ne 5 ]; then
    echo "Usage: $0 <start> <increment> <num_fid_iterations> <num_der_iterations> <output>"
    exit 1
fi

# Parse command line arguments
fid_increment="$1"
der_increment="$2"
num_fid_iterations="$3"
num_der_iterations="$4"
output="$5"

find_minimum() {
    local num1="$1"
    local num2="$2"

    if [ "$num1" -lt "$num2" ]; then
        echo "$num1"
    else
        echo "$num2"
    fi
}


# Define a function to run piFromPDNew.py
run_pi() {
    local param="$1"
    local increment="$2"
    local num_iterations="$3"
    local max_iterations="$4"
    local start=0
    local ranges=()

    # Use a for loop to generate the ranges and append them to the array
    for ((i = 1; i <= num_iterations; i++)); do
        end=$((start + increment))
        end=$(find_minimum "$end" "$max_iterations")
        ranges+=("$param $start $end $output") 
        start=$end
    done

    # Return the ranges array
    echo "${ranges[@]}"
}

# Define an array to store the ranges
all_ranges=()

# Call the run_pi function for different parameters and append the result to the ranges array
all_ranges+=($(run_pi "fiducial" "$fid_increment" "$num_fid_iterations" "10000"))
all_ranges+=($(run_pi "Om_m" "$der_increment" "$num_der_iterations" "500"))
all_ranges+=($(run_pi "Om_p" "$der_increment" "$num_der_iterations" "500"))
all_ranges+=($(run_pi "s8_m" "$der_increment" "$num_der_iterations" "500"))
all_ranges+=($(run_pi "s8_p" "$der_increment" "$num_der_iterations" "500"))

# Loop through the ranges and execute the commands
for ((i = 0; i < ${#all_ranges[@]}; i+=4)); do
    range_args="${all_ranges[$i]} ${all_ranges[$i+1]} ${all_ranges[$i+2]} ${all_ranges[$i+3]}"
    # echo "ranges = $range_args"
    IFS=' ' read -r param start end output <<< "$range_args" 
    # echo "param = $param"
    # echo "start = $start"
    # echo "end = $end"
    # echo "output = $output"
    python3 /projects/0/gusr0688/vk/topofisher/piFromPDNew.py "$param" "$start" "$end" "$output" &
done

# Wait for all background jobs to finish
wait