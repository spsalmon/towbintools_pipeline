#!/bin/bash

#SBATCH -J convert_squid
#SBATCH -o ./sbatch_output/convert_squid.out
#SBATCH -e ./sbatch_output/convert_squid.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=32GB

~/.local/bin/micromamba run -n towbintools python3 convert_squid_experiment.py