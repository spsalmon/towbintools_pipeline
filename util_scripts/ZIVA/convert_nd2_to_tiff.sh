#!/bin/bash
#SBATCH -J convert
#SBATCH -o convert.out
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250929_161842_172_ZIVA_60x_443_stardist_training/raw_nd2_part_4"
OUTPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250929_161842_172_ZIVA_60x_443_stardist_training/raw_part4"
~/.local/bin/micromamba run -n towbintools python3 convert_nd2_to_tiff.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
