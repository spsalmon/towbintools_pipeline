#!/bin/bash

# Run everything from the repo root, regardless of where this is called from.
cd "$(dirname "$0")/.."

# Function to check for git updates
check_git_updates() {
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo "Not in a git repository. Skipping update check."
        return 0
    fi

    # Fetch latest changes from remote without merging
    echo "Checking for updates..."
    git fetch --quiet

    # Get current and remote commit hashes; no upstream means nothing to compare
    local_commit=$(git rev-parse HEAD)
    if ! remote_commit=$(git rev-parse @{u} 2>/dev/null); then
        echo "No remote tracking branch found. Skipping update check."
        return 0
    fi

    # Compare commits
    if [ "$local_commit" != "$remote_commit" ]; then
        echo "A newer version is available!"
        echo "Current commit: ${local_commit:0:8}"
        echo "Latest commit:  ${remote_commit:0:8}"
        echo

        read -p "Would you like to update to the latest version? (y/n): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Updating to the pipeline to the latest version..."
            echo "This will not update the environment, for that, run the scripts/update_pipeline.sh script."
            bash scripts/update_pipeline.sh --pipeline-only
            echo "Pipeline updated successfully! Please restart the script."
        else
            echo "Continuing with current version..."
        fi
    else
        echo "Already up to date!"
    fi
}

# Check for updates
check_git_updates

# if the folder sbatch_output does not exist, create it
if [ ! -d "sbatch_output" ]; then
    mkdir sbatch_output
fi

# if the folder temp_files does not exist, create it
if [ ! -d "temp_files" ]; then
    mkdir temp_files
fi

# Find the config among the forwarded arguments (same default as the sbatch
# script), so we can derive the outer job's sbatch resources from it. Read by
# index rather than shift, so "$@" stays intact for forwarding below.
CONFIG_FILE="./defaults/config/config.yaml"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        -c|--config)
            if (( i + 1 >= ${#args[@]} )); then
                echo "Error: ${args[$i]} requires a value" >&2
                exit 1
            fi
            CONFIG_FILE="${args[$((i + 1))]}"
            ;;
    esac
done

# Config-drive the outer job's sbatch resources from sbatch_init. If this yields
# nothing (e.g. non-slurm config), sbatch falls back to the header in the script.
SBATCH_INIT_FLAGS=$(~/.local/bin/micromamba run -n towbintools python3 -m towbintools_pipeline.run_params --sbatch-init -c "$CONFIG_FILE" 2>/dev/null)

# Pass the resource flags and the forwarded arguments to the SBATCH script.
sbatch $SBATCH_INIT_FLAGS scripts/_sbatch_pipeline.sh "$@"
