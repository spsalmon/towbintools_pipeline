#!/bin/bash

#SBATCH -J preprocess
#SBATCH -o preprocess.out
#SBATCH -e preprocess.err
#SBATCH -c 32
#SBATCH -t 1:00:00
#SBATCH --mem=32GB

DATABASE_PATH="/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/datasets/brightfield_finetuning/body/"
PREPROCESSING_TYPE="binarize"  # Change this to the desired preprocessing type


# Run the Python script with the specified or default configuration file
~/.local/bin/micromamba run -n towbintools python3 preprocess_masks.py --database_path "$DATABASE_PATH" --preprocessing_type "$PREPROCESSING_TYPE"
