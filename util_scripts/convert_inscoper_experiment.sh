#!/bin/bash

ulimit -s unlimited  # Stack size
export ARG_MAX=$(getconf ARG_MAX)  # Check current limit

EXPERIMENT_DIR="/mnt/towbin.data/shared/spsalmon/20250404_ZIVA_40x_397_405_yap_dynamics"

declare -A FOLDER_MAPPING
FOLDER_MAPPING["images"]="raw"
FOLDER_MAPPING["images_1"]="raw_zstack"

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory $EXPERIMENT_DIR does not exist"
    exit 1
fi

echo "Starting file reorganization in: $EXPERIMENT_DIR"

for images_dir in "$EXPERIMENT_DIR"/images*; do
    [ ! -d "$images_dir" ] && continue

    folder_name=$(basename "$images_dir")
    target_name="${FOLDER_MAPPING[$folder_name]}"

    if [ -z "$target_name" ]; then
        echo "Warning: No mapping defined for $folder_name, skipping..."
        continue
    fi

    raw_data_path="$images_dir/RAW_DATA"
    if [ ! -d "$raw_data_path" ]; then
        echo "Warning: $raw_data_path not found, skipping..."
        continue
    fi

    echo "Processing: $folder_name/RAW_DATA/ -> $target_name/"

    file_count=$(find "$raw_data_path" -type f | wc -l)
    echo "Found $file_count files to move"

    target_dir="$EXPERIMENT_DIR/$target_name"
    mkdir -p "$target_dir"

    if [ $file_count -gt 0 ]; then
        echo "Moving files..."

        metadata_dir="${target_dir}_metadata"
        mkdir -p "$metadata_dir"

        # Method 1: Use cd and move with filtering
        if (cd "$raw_data_path" &&
            # Move metadata files first
            [ -f "metadata.companion.ome" ] && mv metadata.companion.ome "$metadata_dir/" 2>/dev/null
            mv *.json "$metadata_dir/" 2>/dev/null
            # Move everything else
            mv ./* "$target_dir/" 2>/dev/null
        ); then
            echo "✓ Successfully moved files to $target_dir"
        else
            echo "✗ Error moving files from $raw_data_path"
            continue
        fi

        metadata_count=$(find "$metadata_dir" -type f | wc -l)
        if [ $metadata_count -gt 0 ]; then
            echo "✓ Moved $metadata_count metadata files to ${target_name}_metadata/"
        else
            rmdir "$metadata_dir" 2>/dev/null
        fi

        rmdir "$raw_data_path" 2>/dev/null

        if [ -z "$(ls -A "$images_dir" 2>/dev/null)" ]; then
            rmdir "$images_dir"
            echo "✓ Removed empty directory $images_dir"
        fi
    fi
done
