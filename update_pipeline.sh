#!/bin/bash

# Check if "pipeline only" argument is provided
PIPELINE_ONLY=false
if [[ "$1" == "pipeline-only" || "$1" == "--pipeline-only" ]]; then
    PIPELINE_ONLY=true
    echo "Skipping environment updates, updating pipeline only."
fi

# Update the pipeline
git fetch origin
git checkout main
git reset --hard origin/main

# Skip environment updates if pipeline-only mode is enabled
if [[ "$PIPELINE_ONLY" == false ]]; then
    echo "Updating environment..."

    # Update micromamba
    ~/.local/bin/micromamba self-update

    # Update conda packages using the lock file.
    ~/.local/bin/micromamba run -n towbintools conda-lock install --name towbintools ./requirements/conda-lock.yml

    # Add the environment to the jupyter notebook kernel
    ~/.local/bin/micromamba run -n towbintools python -m ipykernel install --user --name=towbintools

#     # Explicitly install pip packages from the rendered lock file.
#     # conda-lock install does not reliably update pip packages in existing environments,
#     # so we extract the pip section and run pip directly with the exact locked versions.
#     TEMP_PIP_REQS=$(mktemp /tmp/pip_requirements_XXXXXX.txt)
#     ~/.local/bin/micromamba run -n towbintools python3 -c "
# import yaml
# with open('./requirements/conda-linux-64.lock.yml') as f:
#     data = yaml.safe_load(f)
# for dep in data.get('dependencies', []):
#     if isinstance(dep, dict) and 'pip' in dep:
#         print('\n'.join(dep['pip']))
#         break
# " > "$TEMP_PIP_REQS"
#     ~/.local/bin/micromamba run -n towbintools pip install --require-hashes --no-deps -r "$TEMP_PIP_REQS"
#     rm -f "$TEMP_PIP_REQS"
fi
