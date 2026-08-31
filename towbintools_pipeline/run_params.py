"""Emit run parameters for the shell launchers, derived from the pipeline
config so the launch scripts stay generic. Currently prints the sbatch CLI
flags for the outer/orchestrator job (from sbatch_init).
"""

import argparse

import yaml

from towbintools_pipeline.utils import build_resource_directives
from towbintools_pipeline.utils import merge_slurm_config
from towbintools_pipeline.utils import resolve_init_slurm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    parser.add_argument(
        "--sbatch-init",
        action="store_true",
        help="Print the sbatch CLI flags for the outer/orchestrator job",
    )
    return parser.parse_args()


def sbatch_init_flags(config):
    init = resolve_init_slurm(config)
    return build_resource_directives(
        init.get("sbatch_cpus"),
        init.get("sbatch_time"),
        init.get("sbatch_memory"),
        init.get("sbatch_gpus"),
        init.get("sbatch_extra_options"),
    )


def main():
    args = get_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = merge_slurm_config(config, args.config)

    if args.sbatch_init:
        print(" ".join(sbatch_init_flags(config)))


if __name__ == "__main__":
    main()
