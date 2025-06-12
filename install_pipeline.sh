#!/bin/bash

# Update micromamba
~/.local/bin/micromamba self-update

# Install dependencies using the lock file
~/.local/bin/micromamba create -n towbintools -f ./requirements/conda-lock.yaml -y

# Add the environment to the jupyter notebook kernel
~/.local/bin/micromamba run -n towbintools python -m ipykernel install --user --name=towbintools

mkdir -p ~/towbintools_pipeline/sbatch_output
