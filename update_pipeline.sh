#!/bin/bash

# Update micromamba
~/.local/bin/micromamba self-update

# Update the pipeline
git fetch origin
git reset --hard origin/main

# Update the towbintools package
~/.local/bin/micromamba run -n towbintools pip install --upgrade towbintools
