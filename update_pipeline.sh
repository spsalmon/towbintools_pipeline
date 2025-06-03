#!/bin/bash

# Update micromamba
~/.local/bin/micromamba self-update

# Update the pipeline
git fetch origin
git reset --hard origin/main

# Update the towbintools package
~/.local/bin/micromamba run -n towbintools pip install --upgrade towbintools

# install any new package added
~/.local/bin/micromamba run -n towbintools pip install --upgrade -r ./requirements/requirements_pip.txt
~/.local/bin/micromamba install -n towbintools -f ./requirements/requirements_conda.txt -y
# update all mambda packages
~/.local/bin/micromamba update -n towbintools --all -y
