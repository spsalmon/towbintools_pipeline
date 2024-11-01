#!/bin/bash

# Update micromamba

~/.local/bin/micromamba self-update

# Create environment
~/.local/bin/micromamba create -n towbintools python=3.9

# Activate environment

~/.local/bin/micromamba activate towbintools

# Install imagecodecs

~/.local/bin/micromamba install -c conda-forge imagecodecs

# Install all the cuda stuff

micromamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install ilastik

~/.local/bin/micromamba install -c ilastik-forge ilastik-core

# Install pip in the environment

~/.local/bin/micromamba install pip

# Install the required packages 

python -m pip install -r ~/towbintools_pipeline/requirements.txt

# Add the environment to the jupyter notebook kernel

python -m ipykernel install --user --name=towbintools

# Batshit insane way to make everything work, dirtiest thing ever but it works, don't ask me why

pip uninstall torchvision

micromamba remove pytorch

micromamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

mkdir -p ~/towbintools_pipeline/sbatch_output