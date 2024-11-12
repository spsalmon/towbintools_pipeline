#!/bin/bash

# Update micromamba
~/.local/bin/micromamba self-update

# Create environment
~/.local/bin/micromamba create -n towbintools python=3.9 -y

# Install all the cuda stuff and ilastik

~/.local/bin/micromamba install -n towbintools -c pytorch -c nvidia -c conda-forge -c ilastik-forge pytorch torchvision torchaudio pytorch-cuda=11.8 imagecodecs ilastik-core -y

# Install packages available through conda

~/.local/bin/micromamba install -n towbintools -f ~/towbintools_pipeline/requirements_conda.txt -y

# Install the required packages only available through pip

~/.local/bin/micromamba run -n towbintools python -m pip install -r ~/towbintools_pipeline/requirements_pip.txt

# Add the environment to the jupyter notebook kernel

~/.local/bin/micromamba run -n towbintools python -m ipykernel install --user --name=towbintools

mkdir -p ~/towbintools_pipeline/sbatch_output