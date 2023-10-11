#!/bin/bash

#SBATCH -J pipeline
#SBATCH -o ./sbatch_output/pipeline.out
#SBATCH -e ./sbatch_output/pipeline.err
#SBATCH -c 2
#SBATCH -t 1:00:00
#SBATCH --mem=4GB
#SBATCH --wait

source ~/env_directory/towbintools/bin/activate
python3 pimp_your_pipeline.py
deactivate