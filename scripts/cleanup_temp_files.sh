#!/bin/bash

# Run from the repo root so the relative paths below resolve there.
cd "$(dirname "$0")/.."

rm -rf ./temp_files/*
rm -rf ./sbatch_output/*
