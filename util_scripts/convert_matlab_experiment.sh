#!/bin/bash

#SBATCH -J convert
#SBATCH -o ./sbatch_output/convert.out
#SBATCH -e ./sbatch_output/convert.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=32GB

~/.local/bin/micromamba run -n towbintools python3 convert_matlab_experiment.py
