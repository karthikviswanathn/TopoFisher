#!/bin/bash

# List of parameters to iterate over
params=("Om_m" "Om_p" "s8_m" "s8_p")

# Common arguments for piFromPDManyJobs.sh
increment=1
num_iterations=500
input_directory="/projects/0/gusr0688/pd_sancho"
output_directory="/projects/0/gusr0688/vk/outputs/pers_images_scratch"

# Loop through the parameters and submit jobs
for param in "${params[@]}"; do
    sbatch /projects/0/gusr0688/vk/scripts/PI/piFromPDManyJobs.sh "$param" "$increment" "$num_iterations" "$input_directory" "$output_directory"
done