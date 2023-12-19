#!/bin/bash

# List of parameters to iterate over
params=("fiducial" "Om_m" "Om_p" "s8_m" "s8_p" "h_m" "h_p")

# Common arguments for piFromPDManyJobs.sh
num_iterations=500
#input_directory="/projects/0/gusr0688/jacky/pds_jars"
input_directory="$1"
res_len="$2"
output_directory="/projects/0/gusr0688/vk/outputs/cluster/pis_15"
script_loc="/projects/0/gusr0688/vk/TopoFisher/topofisher/cluster/vectorization/persistence_images/piFromPDManyJobs.sh"
# Loop through the parameters and submit jobs
for param in "${params[@]}"; do
    if [ "$param" == "fiducial" ]; then
        increment=20
    else
        increment=1
    fi
    echo "$param" "$increment"
    sbatch "$script_loc" "$param" "$increment" "$num_iterations" "$input_directory" "$output_directory" "$res_len"
done