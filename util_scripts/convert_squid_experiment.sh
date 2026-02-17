#!/bin/bash

#SBATCH -J convert_squid
#SBATCH -o ../sbatch_output/convert_squid.out
#SBATCH -e ../sbatch_output/convert_squid.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=32GB

INPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250918_SQUID_10x_yapAID_F79G_527_557/squid_raw/"
OUTPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250918_SQUID_10x_yapAID_F79G_527_557/raw/"

~/.local/bin/micromamba run -n towbintools python3 convert_squid_experiment.py
