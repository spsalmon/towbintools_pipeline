#!/bin/bash

# Update the pipeline
git fetch origin
git reset --hard origin/main

# Update the towbintools package
micromamba activate towbintools
pip install --upgrade towbintools