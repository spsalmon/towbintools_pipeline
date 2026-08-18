#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# just in case, reset the repository to the latest version of main
git fetch origin
git checkout main
git reset --hard origin/main

# Update micromamba
~/.local/bin/micromamba self-update

# Build the environment into a fresh prefix and switch the `towbintools` symlink
# over to it. See requirements/build_env.sh for why we never mutate the live env.
bash ./requirements/build_env.sh

mkdir -p ./sbatch_output
