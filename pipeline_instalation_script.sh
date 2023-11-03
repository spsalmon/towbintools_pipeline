#!/bin/bash

# Update micromamba

micromamba self-update

# Create environment

micromamba create -n towbintools --override-channels -c pytorch -c ilastik-forge -c conda-forge ilastik

# Activate environment

micromamba activate towbintools

# Install pip in the environment

micromamba install pip

# Install the required packages 

python -m pip install -r ~/towbintools_pipeline/requirements.txt

# Add the environment to the jupyter notebook kernel

python -m ipykernel install --user --name=towbintools

mkdir -p ~/towbintools_pipeline/sbatch_output