# Docs rewrite — things to cover

Running notes of things to include when the docs (README + book) are rewritten
at the end of the refactor. Add to this as we change things; don't edit the
docs piecemeal in the meantime.

## Repo structure
- `towbintools_pipeline/` = core pipeline package (`python -m
  towbintools_pipeline...`). `defaults/` = bundled `config/` + `models/`
  (fallbacks/examples). `scripts/` = automation bash. `tools/` = one-off
  data-conversion helpers (auxiliary). `examples/custom_scripts/` = user
  extension-point templates. `training/` and `gui/` are separate auxiliary
  trees. Add a repo map to the README.
- Run/install now via scripts/: `bash scripts/run_pipeline.sh`,
  `bash scripts/install_pipeline.sh`.

## Installation / environment
- Local install without micromamba, any OS: `conda env create -f
  requirements/environment_local.yml`, then `conda activate towbintools_local`.
- `lxml` is needed to read OME-TIFF metadata cleanly (otherwise a warning).
- Keep the existing cluster path documented too (micromamba + conda-lock +
  `scripts/install_pipeline.sh`) — install logic unchanged, only moved to
  scripts/.
- `pip install -e ".[dev]"` (via pyproject.toml) installs the pipeline as a
  package so it runs from anywhere (no repo-root / PYTHONPATH). Local install is
  two steps from the repo root: `conda env create -f
  requirements/environment_local.yml`, then `pip install -e ".[dev]"`. (conda
  resolves a pip `-e .` relative to the yml's dir, so we don't put it in the yml.)
- Explain the dependency layering:
  1. pyproject.toml = what the pipeline needs (abstract deps, single source).
  2. environment*.yml = how to build an env (conda-vs-pip delivery choice).
  3. conda-lock.yml / conda-linux-64.lock = exact pinned solve (cluster).

## Deferred cleanup (engineering, not just docs)
- Consolidate the CLUSTER `environment.yml` to source deps from pyproject
  (pip section `- .`), so the dep list is not duplicated; then regenerate the
  lock. Needs deliberate testing on the cluster — do it in the packaging
  milestone, not before.

## Running the pipeline
- New `backend` config option: `slurm` (default, submits jobs) vs `local` (runs
  in-process, no slurm/micromamba).
- Local run, from the repo root:
  `python -m towbintools_pipeline.init_pipeline -c <config> --temp_dir <dir>`.
- For now the pipeline must be launched from the repo root (module invocation);
  this goes away once it is a proper installed package with an entry point.
- `experiment_dir` can be given with `--experiment_dir` (overrides the config).
- Temp/output location precedence: `--temp_dir` flag > `temp_dir` config key >
  default `<experiment_dir>/temp_files`. Nothing is written inside the repo by
  default (outputs already live under `experiment_dir`). Note the outer launch
  script still makes a repo-root `sbatch_output/` — part of the
  `_sbatch_pipeline.sh` follow-up below.

## Testing
- `python -m pytest tests/ -v` runs the local-backend smoke test (synthetic
  images, no bundled data). Document how to add more tests.

## Config
- Document `backend` (`slurm` vs `local`).
- `n_jobs` (main config, all backends) = worker compute parallelism (joblib
  `--n_jobs`). Falls back to `sbatch_cpus` and vice versa, so setting one is
  enough.
- SLURM resources live in a separate file `configs/slurm_config.yaml`, pointed
  to by `slurm_config:` in the main config (relative to it) and merged in at
  startup for the slurm backend only. Inline `sbatch_*` keys override the file.
- Each standard sbatch directive (`-c`, `-t`, `--mem`, gpu gres) is emitted only
  when set, so one can be dropped (e.g. omit `sbatch_memory`, use
  `--mem-per-cpu` instead). `sbatch_extra_options` is a list of raw sbatch
  option strings rendered verbatim as `#SBATCH <option>` lines — cluster-specific
  directives (`--account`, `--mem-per-cpu`, `--partition`, custom gres) are now
  config-only, no edits to `towbintools_pipeline/utils.py`.
- Per-block SLURM resources: the top-level `sbatch_*` keys are the shared
  default for every worker block. Override per block type under
  `sbatch_overrides` (keyed by block name, e.g. `segmentation`), merged over the
  default. The outer/orchestrator job's resources go under `sbatch_init`.
  Per-instance (a specific occurrence) overrides are still future work.
- The outer/orchestrator job's resources come from `sbatch_init`. `run_pipeline.sh`
  turns them into sbatch CLI flags (via `python -m towbintools_pipeline.run_params
  --sbatch-init`) which override `_sbatch_pipeline.sh`'s minimal header. So a new
  cluster is adjusted entirely in the config now — cluster-specific outer
  directives (`--account`, a custom `--gres` like the old `pipelinecapacity`
  throttle, `--mem-per-cpu`) go under `sbatch_init.sbatch_extra_options`.
- `run_pipeline.sh` forwards `-e/--experiment_dir` and `-t/--temp_dir` through to
  the pipeline (previously only `-c`).
- Still hardcoded in `_sbatch_pipeline.sh`: the `micromamba run -n towbintools`
  env name (env-decoupling, separate milestone).
- Temp working dir defaults to in-repo `./temp_files` (gitignored, transient,
  cleared by cleanup_temp_files.sh). Decision: keep this default rather than
  auto-placing it next to the experiment — the durable outputs and backup are
  already external, and `-t <path>` covers the large-experiment / home-quota
  case (e.g. put temp on the data storage). Not worth the bash complexity of
  resolving the experiment dir before Python runs.

## Run backup / provenance
- The pipeline snapshots the config(s) used and a `git_info.txt` (git
  branch/commit/status + interpreter/package versions) into the run's temp dir,
  which syncs into the backup. Done in Python, so local runs are recorded too
  (previously only the sbatch launcher did this, slurm-only).
- The backup lives at `<experiment>/analysis/pipeline_backup/pipeline_<id>/`
  (beside `report/`, not inside it — report holds results, backup holds
  provenance).
- Temp and backup are write-only records: the pipeline reads its config from the
  original path given with `-c`, never from the temp copy. This is why a
  relative `slurm_config:` resolves against the original config's directory. (The
  old launcher copied the config into temp and ran that copy, so the sibling
  `slurm_config.yaml` was silently not found.)

## Config
- `analysis_dir_name` (default `analysis`) is honored everywhere, including the
  prefix-stripping in output naming. The `report/` subfolder name is currently
  fixed (no config key).

## Deferred design / cleanup (later PRs)
- Folder inputs: decouple internal references from the analysis-dir name (today
  the config repeats the prefix, so renaming `analysis_dir_name` forces
  rewriting every reference). Aim to support: (a) absolute path, (b) name-only
  (resolved under the experiment folder), (c) maybe relative. One place holds
  the dir name.
- Default config uses mixed single/double quotes for strings — make consistent
  (or drop unnecessary quotes).

## Known cleanups to mention / finish before docs
- `CustomBuildingBlock.create_command` was broken (missing `config` param) — fix
  and document custom blocks.
- Workers use a bare `import utils` (rely on script dir on sys.path) — revisit
  when packaging.
- The outer orchestrator job `_sbatch_pipeline.sh` still has its own hardcoded
  `#SBATCH` header (`-c 8 -t 12:00:00 --mem=8GB --gres=pipelinecapacity:1`) and a
  hardcoded `micromamba run -n towbintools`. The per-block worker headers are now
  config-driven, but this launch script is not — sbatch reads its `#SBATCH`
  lines before any YAML is loaded, so config-driving it needs a different
  mechanism (generate the script, or pass `sbatch` CLI flags). Separate
  follow-up.
