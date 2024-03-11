#!/bin/bash

#SBATCH -J pipeline
#SBATCH -o ./sbatch_output/time.out
#SBATCH -e ./sbatch_output/time.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=32GB

~/.local/bin/micromamba run -n towbintools python3 extract_time_from_filemap.py