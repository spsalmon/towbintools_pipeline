#!/bin/bash
#SBATCH -J convert
#SBATCH -o convert.out
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/spradhan/20250929_embryogenesis_316_0uM_500uM_P0_20x_Orca_III/raw_nd2"
OUTPUT_DIR="/mnt/towbin.data/shared/spradhan/20250929_embryogenesis_316_0uM_500uM_P0_20x_Orca_III/raw"
~/.local/bin/micromamba run -n towbintools python3 convert_nd2_to_tiff.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
