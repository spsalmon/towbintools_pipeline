#!/bin/bash

# Update micromamba

~/.local/bin/micromamba self-update

# Create environment
~/.local/bin/micromamba create -n towbintools python=3.9

# Activate environment

~/.local/bin/micromamba activate towbintools

# Install all the cuda stuff

micromamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install imagecodecs

~/.local/bin/micromamba install -c conda-forge imagecodecs=2024.1.1

# Install ilastik

~/.local/bin/micromamba install -c ilastik-forge ilastik-core

# Install pip in the environment

~/.local/bin/micromamba install pip

# Install the required packages 

python -m pip install -r ~/towbintools_pipeline/requirements.txt

# Add the environment to the jupyter notebook kernel

python -m ipykernel install --user --name=towbintools

mkdir -p ~/towbintools_pipeline/sbatch_output