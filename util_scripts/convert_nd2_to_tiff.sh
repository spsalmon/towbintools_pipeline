#!/bin/bash
#SBATCH -J convert_nd2
#SBATCH -o ../sbatch_output/convert_nd2.out
#SBATCH -e ../sbatch_output/convert_nd2.err
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/raw_movies_nd2/"
OUTPUT_DIR="/mnt/towbin.data/shared/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/raw_movies/"
NJOBS=2
~/.local/bin/micromamba run -n towbintools python3 convert_nd2_to_tiff.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --njobs "$NJOBS"
