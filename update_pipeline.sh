#!/bin/bash

# Update micromamba
~/.local/bin/micromamba self-update

# Update the pipeline
git fetch origin
git reset --hard origin/main

# Recreate the environment using the lock file
~/.local/bin/micromamba create -n towbintools -f ./requirements/conda-lock.yaml -y

# Add the environment to the jupyter notebook kernel
~/.local/bin/micromamba run -n towbintools python -m ipykernel install --user --name=towbintools
