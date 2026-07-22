#!/bin/bash
#SBATCH -J pipeline
#SBATCH -o ./sbatch_output/pipeline-%j.out
#SBATCH -e ./sbatch_output/pipeline-%j.err
# Resource directives (-c/-t/--mem/--gres/--account/...) are passed on the
# sbatch command line by run_pipeline.sh, derived from the config's sbatch_init.
OMP_NUM_THREADS=1
# Default configuration file
DEFAULT_CONFIG_FILE="./defaults/config/config.yaml"
CONFIG_FILE="$DEFAULT_CONFIG_FILE"
TEMP_DIR="./temp_files"
EXPERIMENT_DIR=""
# Function to show usage
usage() {
    echo "Usage: $0 [-c <config>] [-e <experiment_dir>] [-t <temp_dir>]" >&2
    exit 1
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
        -e|--experiment_dir)
        EXPERIMENT_DIR="$2"
        shift
        shift
        ;;
        -t|--temp_dir)
        TEMP_DIR="$2"
        shift
        shift
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
# Move this job's logs in with the per-block ones and update SLURM to write
# there going forward. Named like a block so the combined pipeline-<id>.out/.err
# the linker writes to the parent picks them up as the first section. The dir is
# created here because Python only makes it later.
LOG_DIR="$TEMP_DIR/sbatch_output"
mkdir -p "$LOG_DIR"
mv "./sbatch_output/pipeline-${SLURM_JOB_ID}.out" "$LOG_DIR/init-${SLURM_JOB_ID}.out"
mv "./sbatch_output/pipeline-${SLURM_JOB_ID}.err" "$LOG_DIR/init-${SLURM_JOB_ID}.err"
scontrol update JobId=$SLURM_JOB_ID StdOut="$LOG_DIR/init-${SLURM_JOB_ID}.out" StdErr="$LOG_DIR/init-${SLURM_JOB_ID}.err"
# Run the pipeline on the ORIGINAL config so relative paths (e.g. the SLURM
# config) resolve; temp/backup stay a write-only record of the run.
~/.local/bin/micromamba run -n towbintools python3 -m towbintools_pipeline.init_pipeline -c "$CONFIG_FILE" --temp_dir "$TEMP_DIR" ${EXPERIMENT_DIR:+-e "$EXPERIMENT_DIR"}
