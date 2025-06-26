#!/bin/bash

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

    # Get current and remote commit hashes
    local_commit=$(git rev-parse HEAD)
    remote_commit=$(git rev-parse @{u} 2>/dev/null)

    # Check if remote tracking branch exists
    if [ $? -ne 0 ]; then
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
            echo "This will not update the environment, for that, run the update_pipeline.sh script."
            bash update_pipeline.sh --pipeline-only
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

# Pass command line arguments to the SBATCH script
sbatch _sbatch_pipeline.sh "$@"
