# Docs rewrite — things to cover

Running notes of things to include when the docs (README + book) are rewritten
at the end of the refactor. Add to this as we change things; don't edit the
docs piecemeal in the meantime.

## Installation / environment
- Local install without micromamba, any OS: `conda env create -f
  requirements/environment_local.yml`, then `conda activate towbintools_local`.
- `lxml` is needed to read OME-TIFF metadata cleanly (otherwise a warning).
- Keep the existing cluster path documented too (micromamba + conda-lock +
  `install_pipeline.sh`) — it is unchanged.
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
  `python -m pipeline_scripts.init_pipeline -c <config> --temp_dir <dir>`.
- For now the pipeline must be launched from the repo root (module invocation);
  this goes away once it is a proper installed package with an entry point.

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
  config-only, no edits to `pipeline_scripts/utils.py`.
- Per-block SLURM resources are still future work (currently one resource set
  for all worker jobs).

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
