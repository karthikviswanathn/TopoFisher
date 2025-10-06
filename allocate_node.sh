#!/bin/bash

# Default time is 1 hour if no argument is provided
HOURS=${1:-1}

srun --partition=dev-g \
     --nodes=1 \
     --gpus-per-node=1 \
     --time=$(printf "%02d:00:00" "$HOURS") \
     --account=project_465001340 \
     --pty bash -c "
        module --force purge
        module load LUMI/24.03 partition/G Boost/1.83.0-cpeGNU-24.03 oneTBB/2021.13.0-cpeGNU-24.03
        ml use /appl/local/csc/modulefiles/
        module load pytorch/2.7

        # Configure Singularity container for multipers (requires TBB library)
        export SING_FLAGS=\"\$SING_FLAGS -B /users/viswanat/.local/lib\"
        export SINGULARITYENV_LD_LIBRARY_PATH=\"/users/viswanat/.local/lib:\$LD_LIBRARY_PATH\"

        exec bash
     "