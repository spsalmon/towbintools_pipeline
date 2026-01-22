#!/bin/bash
#SBATCH -J convert
#SBATCH -o convert.out
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/spsalmon/20260116_153855_560_ZIVA_20x_125_619_pumping_test/raw_nd2/"
OUTPUT_DIR="/mnt/towbin.data/shared/spsalmon/20260116_153855_560_ZIVA_20x_125_619_pumping_test/raw/"
~/.local/bin/micromamba run -n towbintools python3 convert_nd2_to_tiff.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
