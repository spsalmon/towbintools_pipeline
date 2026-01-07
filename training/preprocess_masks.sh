#!/bin/bash

#SBATCH -J preprocess
#SBATCH -o preprocess.out
#SBATCH -e preprocess.err
#SBATCH -c 32
#SBATCH -t 1:00:00
#SBATCH --mem=32GB

DATABASE_PATH="/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/datasets/60x_brightfield/brightfield/"
PREPROCESSING_TYPE="binarize"  # Change this to the desired preprocessing type
KEEP_ONLY_BIGGEST_OBJECT=true

if [ "$KEEP_ONLY_BIGGEST_OBJECT" = true ]; then
    ~/.local/bin/micromamba run -n towbintools python3 preprocess_masks.py --database_path "$DATABASE_PATH" --preprocessing_type "$PREPROCESSING_TYPE" --keep_only_biggest_object
else
    ~/.local/bin/micromamba run -n towbintools python3 preprocess_masks.py --database_path "$DATABASE_PATH" --preprocessing_type "$PREPROCESSING_TYPE"
fi
