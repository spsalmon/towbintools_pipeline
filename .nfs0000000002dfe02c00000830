#!/bin/bash

# Update micromamba
~/.local/bin/micromamba self-update

# Create environment
~/.local/bin/micromamba create -n towbintools python=3.9 -y

# # Install all the cuda stuff

~/.local/bin/micromamba install -n towbintools -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 "numpy<1.20" imagecodecs -y

# Install the required packages 

~/.local/bin/micromamba run -n towbintools python -m pip install -r ~/towbintools_pipeline/requirements.txt

# Add the environment to the jupyter notebook kernel

~/.local/bin/micromamba run -n towbintools python -m ipykernel install --user --name=towbintools

# Install ilastik

~/.local/bin/micromamba install -n towbintools -c ilastik-forge ilastik-core -y

# # Batshit insane way to make everything work, dirtiest thing ever but it works, don't ask me why
# # Apparently not needed anymore

# ~/.local/bin/micromamba run -n towbintools python -m pip uninstall torchvision

# ~/.local/bin/micromamba run -n towbintools python -m pip uninstall pytorch

# ~/.local/bin/micromamba install -n towbintools -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 "numpy<1.20" -y

mkdir -p ~/towbintools_pipeline/sbatch_output