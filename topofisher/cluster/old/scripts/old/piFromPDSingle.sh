#!/bin/bash

#SBATCH -n 64  
#SBATCH -t 01:00:00

#SBATCH --job-name=vectorizePD
#SBATCH --output=/projects/0/gusr0688/vk/cmd_outputs/piFromPDSingle.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.viswanathan@uva.nl


module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load SciPy-bundle/2021.05-foss-2021a

echo "Loaded libraries. Running the code now" 

# Check if the number of arguments is correct
if [ $# -ne 1 ]; then
    echo $#
    echo "Usage: $0 <directory>"
    exit 1
fi


# Directory containing the files you want to read
directory="$1"

# Loop through all files in the directory
for file in "$directory"/*; do
    python3 /projects/0/gusr0688/vk/topofisher/piFromPDSingle.py $file &
done

# Wait for all background jobs to finish
wait






