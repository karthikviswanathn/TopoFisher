#!/bin/bash

# Default time is 1 hour if no argument is provided
HOURS=${1:-1}

srun --partition=small-g \
     --nodes=1 \
     --gpus-per-node=1 \
     --time=$(printf "%02d:00:00" "$HOURS") \
     --account=project_465002390 \
     --pty bash -c "
        module --force purge
        module load LUMI/24.03 partition/G Boost/1.83.0-cpeGNU-24.03 oneTBB/2021.13.0-cpeGNU-24.03
        ml use /appl/local/csc/modulefiles/
        module load pytorch/2.7

        # Configure Singularity container for multipers (requires TBB library)
        export SING_FLAGS=\"\$SING_FLAGS -B /users/viswanat/.local/lib\"
        export SINGULARITYENV_LD_LIBRARY_PATH=\"/users/viswanat/.local/lib:\$LD_LIBRARY_PATH\"

        # Set temp directory for HIP/ROCm to avoid Singularity permission issues
        export TMPDIR=\"/pfs/lustrep1/projappl/project_465002390/fair_stuff/TopoFisher/ai-code/tmp/hip\"
        export TEMP=\$TMPDIR
        export TMP=\$TMPDIR
        export HIP_TEMP_DIR=\$TMPDIR
        # Ensure Singularity container also uses our temp directory
        export SINGULARITYENV_TMPDIR=\$TMPDIR
        export SINGULARITYENV_TEMP=\$TMPDIR
        export SINGULARITYENV_TMP=\$TMPDIR

        exec bash
     "