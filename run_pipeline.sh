#!/bin/bash

# if the folder sbatch_output does not exist, create it
if [ ! -d "sbatch_output" ]; then
    mkdir sbatch_output
fi
# if the folder temp_files does not exist, create it
if [ ! -d "temp_files" ]; then
    mkdir temp_files
fi

# Pass command line arguments to the SBATCH script
sbatch _sbatch_pipeline.sh "$@"