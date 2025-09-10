#!/bin/bash
#SBATCH -J best_plane
#SBATCH -o best_plane.out
#SBATCH -c 64
#SBATCH -t 12:00:00
#SBATCH --mem=64G

# TO BE CHANGED TO YOUR TRAINING AND VALIDATION SET DIRECTORIES

INPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250904_170830_573_ZIVA_40x_397_405_yap_gfp/raw/"
OUTPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250904_170830_573_ZIVA_40x_397_405_yap_gfp/raw_best_plane/"
~/.local/bin/micromamba run -n towbintools python3 postprocess_autofocus.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
