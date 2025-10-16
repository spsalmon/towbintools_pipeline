#!/bin/bash
#SBATCH -J convert
#SBATCH -o convert.out
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/spsalmon/20251014_150718_923_ZIVA_60x_443_additional_stardist_training_data/part2/raw_nd2"
OUTPUT_DIR="/mnt/towbin.data/shared/spsalmon/20251014_150718_923_ZIVA_60x_443_additional_stardist_training_data/part2/raw"
~/.local/bin/micromamba run -n towbintools python3 convert_nd2_to_tiff.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
