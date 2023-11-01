#!/bin/bash

#SBATCH -J pipeline
#SBATCH -o ./sbatch_output/pipeline.out
#SBATCH -e ./sbatch_output/pipeline.err
#SBATCH -c 2
#SBATCH -t 96:00:00
#SBATCH --mem=4GB

~/.local/bin/micromamba run -n towbintools python3 pimp_your_pipeline.py