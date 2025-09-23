#!/bin/bash
#SBATCH -J convert
#SBATCH -o convert.out
#SBATCH -c 64
#SBATCH -t 12:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250904_170830_573_ZIVA_40x_397_405_yap_gfp/raw_nd2/"
OUTPUT_DIR="/mnt/towbin.data/shared/spsalmon/20250904_170830_573_ZIVA_40x_397_405_yap_gfp/raw/"
~/.local/bin/micromamba run -n towbintools python3 convert_nd2_to_tiff.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
