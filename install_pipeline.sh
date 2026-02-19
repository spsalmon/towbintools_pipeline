#!/bin/bash

# just in case, reset the repository to the latest version of main
git fetch origin
git checkout main
git reset --hard origin/main

# Update micromamba
~/.local/bin/micromamba self-update

# Install dependencies using the lock file
~/.local/bin/micromamba create -n towbintools -f ./requirements/conda-lock.yml -y

# Add the environment to the jupyter notebook kernel
~/.local/bin/micromamba run -n towbintools python -m ipykernel install --user --name=towbintools

mkdir -p ./sbatch_output
