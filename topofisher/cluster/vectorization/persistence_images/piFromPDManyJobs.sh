#!/bin/bash

#SBATCH -n 128
#SBATCH -t 01:00:00
#SBATCH --job-name=vectorizePD
#SBATCH --output=/projects/0/gusr0688/vk/cmd_outputs/piFromPD_R_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vkarthik095@gmail.com

module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load SciPy-bundle/2021.05-foss-2021a

echo "Loaded libraries. Copying to scratch now" 

# Check if the number of arguments is correct
if [ $# -ne 6 ]; then
    echo "Usage: $0 <start> <increment> <num_fid_iterations> <input_directory> <output>"
    exit 1
fi

# Parse command line arguments
parameter_name="$1"
increment="$2" # Number of PDs per thread
num_iterations="$3" # Number of parallel threads to run
input_directory="$4"
input_directory_name=$(basename "$input_directory")
output_directory="$5"
res_len="$6"

cp -r "$input_directory" "$TMPDIR"
echo "Copied files to scratch. Extracting now"

# Loop through the .tar.gz files and extract them in place
for file in "$TMPDIR"/"$input_directory_name"/"$parameter_name"_*.tar.gz; do
    # echo "Extracting $file..."
    tar -xzf "$file" -C "$TMPDIR"/"$input_directory_name" &
done
wait
echo "Extracting complete..."
# Create output directory on scratch
output_directory_scratch="$TMPDIR"/output_vecs_"$parameter_name"
mkdir "$output_directory_scratch"

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
        ranges+=("$param $start $end "$TMPDIR"/"$input_directory_name" "$output_directory_scratch"") 
        start=$end
    done

    # Return the ranges array
    echo "${ranges[@]}"
}

# Define an array to store the ranges
all_ranges=()
if [ "$parameter_name" = "fiducial" ]; then
    max_iterations="10000"
else
    max_iterations="500"
fi
# Call the run_pi function for different parameters and append the result to the ranges array
all_ranges+=($(run_pi "$parameter_name" "$increment" "$num_iterations" "$max_iterations"))

pyfile_loc="/projects/0/gusr0688/vk/TopoFisher/topofisher/cluster/vectorization/persistence_images/piFromPDNew.py"

# Loop through the ranges and execute the commands
for ((i = 0; i < ${#all_ranges[@]}; i+=5)); do
    range_args="${all_ranges[$i]} ${all_ranges[$i+1]} ${all_ranges[$i+2]} ${all_ranges[$i+3]} ${all_ranges[$i+4]}"
    IFS=' ' read -r param start end input output <<< "$range_args" 
    python3 "$pyfile_loc" "$param" "$start" "$end" "$input" "$output" "$res_len" &
done
# Wait for all background jobs to finish
wait
cp -r "$output_directory_scratch" "$output_directory"
