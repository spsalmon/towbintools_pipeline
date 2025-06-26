#!/bin/bash

# Check if "pipeline only" argument is provided
PIPELINE_ONLY=false
if [[ "$1" == "pipeline-only" || "$1" == "--pipeline-only" ]]; then
    PIPELINE_ONLY=true
    echo "Skipping environment updates, updating pipeline only."
fi

# Update the pipeline
git fetch origin
git reset --hard origin/main

# Skip environment updates if pipeline-only mode is enabled
if [[ "$PIPELINE_ONLY" == false ]]; then
    echo "Updating environment..."

    # Update micromamba
    ~/.local/bin/micromamba self-update

    # Recreate the environment using the lock file
    ~/.local/bin/micromamba create -n towbintools -f ./requirements/conda-lock.yaml -y

    # Add the environment to the jupyter notebook kernel
    ~/.local/bin/micromamba run -n towbintools python -m ipykernel install --user --name=towbintools
fi
