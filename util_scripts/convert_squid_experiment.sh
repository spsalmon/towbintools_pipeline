#!/bin/bash

#SBATCH -J convert_squid
#SBATCH -o ../sbatch_output/convert_squid.out
#SBATCH -e ../sbatch_output/convert_squid.err
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH --mem=16GB

INPUT_DIR="/mnt/towbin.data/shared/nschoonjans/20260120_nathan_eat_fun/squid_raw/"
OUTPUT_DIR="/mnt/towbin.data/shared/nschoonjans/20260120_nathan_eat_fun/raw/"

~/.local/bin/micromamba run -n towbintools python3 convert_squid_experiment.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR
