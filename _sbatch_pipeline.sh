#!/bin/bash
#SBATCH -J pipeline
#SBATCH -o ./sbatch_output/pipeline-%j.out
#SBATCH -e ./sbatch_output/pipeline-%j.err
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --mem=8GB
#SBATCH --gres=pipelinecapacity:1
OMP_NUM_THREADS=1
# Default configuration file
DEFAULT_CONFIG_FILE="./configs/config.yaml"
CONFIG_FILE="$DEFAULT_CONFIG_FILE"
TEMP_DIR="./temp_files"
# Function to show usage
usage() {
    echo "Usage: $0 [-c <config_file> | --config <config_file>]" >&2
    exit 1
}
# function to save version control info to a file in the backup directory
save_version_control_info() {
    local backup_dir="$1"
    local git_info_file="$backup_dir/git_info.txt"
    {
        echo "Git Branch: $(git rev-parse --abbrev-ref HEAD)"
        echo "Git Commit: $(git rev-parse HEAD)"
        echo "Git Status:"
        git status
        echo ""
        echo "Python Version: $(~/.local/bin/micromamba run -n towbintools python3 --version 2>&1)"
        echo "towbintools Version: $(~/.local/bin/micromamba run -n towbintools python3 -c "import towbintools; print(towbintools.__version__)" 2>&1)"
    } > "$git_info_file"
}
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--config)
        CONFIG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        usage
        ;;
    esac
done
# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE" >&2
    exit 1
fi
# Get the number of the slurm job
SLURM_JOB_ID=${SLURM_JOB_ID:-0}
# Create a temporary directory for the job
TEMP_DIR="$TEMP_DIR/pipeline_$SLURM_JOB_ID"
mkdir -p "$TEMP_DIR"
# Copy the configuration file to the temporary directory
cp "$CONFIG_FILE" "$TEMP_DIR"
# Save version control info to the backup directory
save_version_control_info "$TEMP_DIR"
# Move logs to temp dir and update SLURM to write there going forward
mv "./sbatch_output/pipeline-${SLURM_JOB_ID}.out" "$TEMP_DIR/"
mv "./sbatch_output/pipeline-${SLURM_JOB_ID}.err" "$TEMP_DIR/"
scontrol update JobId=$SLURM_JOB_ID StdOut="$TEMP_DIR/pipeline-${SLURM_JOB_ID}.out" StdErr="$TEMP_DIR/pipeline-${SLURM_JOB_ID}.err"
config_file_name=$(basename "$CONFIG_FILE")
CONFIG_FILE="$TEMP_DIR/$config_file_name"
# Run the Python script with the specified or default configuration file
~/.local/bin/micromamba run -n towbintools python3 -m pipeline_scripts.init_pipeline -c "$CONFIG_FILE" --temp_dir "$TEMP_DIR"
