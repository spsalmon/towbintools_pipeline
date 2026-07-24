#!/bin/bash

# Manual scratch cleanup: clears the DEFAULT temp dir (all pipeline_* runs) and
# the repo-root sbatch_output landing zone. Only the default location -- a custom
# temp_dir is not touched here (the cleanup_on_success config key does clear each
# run's dir wherever temp_dir points, but only on success).
cd "$(dirname "$0")/.."

rm -rf ./temp_files/*
rm -rf ./sbatch_output/*
