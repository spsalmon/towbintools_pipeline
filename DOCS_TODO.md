# Docs rewrite — things to cover

Running notes of things to include when the docs (README + book) are rewritten
at the end of the refactor. Add to this as we change things; don't edit the
docs piecemeal in the meantime.

Decisions that cost something go in `TRADEOFFS.md` instead — that file is the
list to hand to a reviewer, this one feeds the docs.

## Repo structure
- `towbintools_pipeline/` = core pipeline package (`python -m
  towbintools_pipeline...`), with `workers/` (one worker per block) and
  `defaults/` = bundled `config/` + `models/` (fallbacks/examples, shipped as
  package data and resolved package-relative). `scripts/` = automation bash.
  `tools/` = one-off data-conversion helpers (auxiliary). `examples/custom_scripts/`
  = user extension-point templates. `training/` and `gui/` are separate auxiliary
  trees. Add a repo map to the README.
- Run/install now via scripts/: `bash scripts/run_pipeline.sh`,
  `bash scripts/install_pipeline.sh`.

## Installation / environment
- Local install without micromamba, any OS: `conda env create -f
  env/environment_local.yml`, then `conda activate towbintools_local`.
- `lxml` is needed to read OME-TIFF metadata cleanly (otherwise a warning).
- Keep the existing cluster path documented too (micromamba + conda-lock +
  `scripts/install_pipeline.sh`) — install logic unchanged, only moved to
  scripts/.
- `pip install -e ".[dev]"` (via pyproject.toml) installs the pipeline as a
  package so it runs from anywhere (no repo-root / PYTHONPATH). Local install is
  two steps from the repo root: `conda env create -f
  env/environment_local.yml`, then `pip install -e ".[dev]"`. (conda
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
- Installed console command (equivalent, works from any cwd once pip-installed):
  `towbintools-pipeline -c <config> --temp_dir <dir>` (entry point ->
  `init_pipeline:main`). The `-m` form stays the fallback and is still what the
  cluster launcher uses until the package is installed there (env consolidation).
- `experiment_dir` can be given with `--experiment_dir` (overrides the config).
- Temp/output location precedence: `--temp_dir` flag > `temp_dir` config key >
  default `<experiment_dir>/temp_files`. Nothing is written inside the repo by
  default (outputs already live under `experiment_dir`). Note the outer launch
  script still makes a repo-root `sbatch_output/` — part of the
  `init_pipeline.sh` follow-up below.

## Which script does what
- `scripts/run_pipeline.sh` — the user entry point, runs on the login node:
  creates the repo-root `sbatch_output/` landing zone, finds the config among the
  arguments, resolves the python launcher, derives the outer job's sbatch flags
  from `sbatch_init`, submits. No longer auto-updates from git; to update, run
  `scripts/update_pipeline.sh` (`--pipeline-only` to skip the env rebuild).
- `scripts/init_pipeline.sh` — the submitted job, named after the module it
  launches: a `#SBATCH` header (which must live in a file for sbatch), the env
  activation, and `"$@"` passed straight through. Nothing else belongs here.
- Everything else is Python. `setup_run_dir()` resolves the per-run directory
  (`pipeline_<jobid>` under slurm, `pipeline_<timestamp>` otherwise) and, under
  slurm, moves the launcher's logs into it — so the temp path has a single
  definition instead of one in bash and one in argparse. Tradeoff: if the env is
  broken and Python never starts, those logs stay in the repo-root
  `sbatch_output/` instead of moving into the run dir.
- Only `batch/` and `sbatch_output/` are slurm-specific; every other write
  (outputs, report, backup, provenance, pickles) happens on both backends.
- `towbintools_pipeline/workers/` holds one worker per block implementation,
  named after its block (`straightening`, `morphology_computation`,
  `segmentation_non_learning`/`segmentation_learning_based`, ...). Each
  `BuildingBlock` stores its `worker_module` and `create_command` runs it with
  `python -m towbintools_pipeline.workers.<name>`, so workers resolve by import
  rather than by a working-directory-relative path. Custom blocks run a
  user-supplied `custom_script_path` (file, `.py` or `.sh`) instead. Generated
  job scripts and logs stay named after the BLOCK (what `sbatch_overrides` keys
  on and the progress log prints); a block can map to several workers.

## Log output
- The run prints a numbered plan of the blocks up front, then a
  `### Starting block 3/12 straightening ###` / `### Finished block ... ###` pair
  around each one, and a closing line. Same on both backends: on slurm the
  linker is appended to the worker's job script, so the finish line and the next
  block's submission land in that block's sbatch `.out` file. A
  `Submitting <block> to slurm ...` line precedes sbatch's own
  "Submitted batch job <id>" so the job id has context.
- Prints that precede a subprocess are flushed, otherwise Python's block
  buffering (stdout redirected to a file under slurm) reorders them after the
  worker's output.
- Slurm log layout: every job writes into `<temp>/pipeline_<id>/sbatch_output/`,
  the outer one as `init_pipeline-<id>.out/.err` (so it sorts first), each block as
  `<block>-<id>.out/.err`. The linker joins them per stream into
  `<temp>/pipeline_<id>/pipeline-<id>.out` (and `.err`) — one file to read the
  whole chain — with a `===== <file> =====` header per section. Originals are
  kept, and the combined files sync into the run backup.
- The combined files are rebuilt at every link, not just at the end, so a run
  that dies mid-chain still leaves a readable log. Their last line says which:
  `PIPELINE FINISHED -- all N blocks completed` only ever gets written by the
  link that found no next block, so anything else (`pipeline still running --
  k/N blocks done so far`) means the run stopped without reaching the end.
- Deliberately NOT a separate end-of-run slurm job: it would sit in the queue,
  so the completion marker could arrive long after the run actually ended, and
  a failed submission would lose the marker entirely. The final link writes it
  instead. The prints it makes are flushed before the concatenation, so they do
  land in the combined log.

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
- Merge rule inside the slurm config: scalar `sbatch_*` keys in a section
  (`sbatch_overrides.<block>`, `sbatch_init`) REPLACE the top-level default,
  while `sbatch_extra_options` entries are APPENDED to it. So cluster-wide
  invariants (`--account`, `--mem-per-cpu`) belong at the top level and always
  apply; a section can add to them but not drop them (move an option down into
  the sections if it must not apply everywhere).
- `sbatch_init` is deliberately NOT a key inside `sbatch_overrides`: the outer
  job is not a building block, it's resolved by a different program at a
  different time (`run_params.py`, from bash, before the pipeline starts), and
  it would collide with the block-name namespace.
- The outer/orchestrator job's resources come from `sbatch_init`. `run_pipeline.sh`
  turns them into sbatch CLI flags (via `python -m towbintools_pipeline.run_params
  --sbatch-init`) which override `init_pipeline.sh`'s minimal header. So a new
  cluster is adjusted entirely in the config now — cluster-specific outer
  directives (`--account`, a custom `--gres` like the old `pipelinecapacity`
  throttle, `--mem-per-cpu`) go under `sbatch_init.sbatch_extra_options`.
- `run_pipeline.sh` forwards `-e/--experiment_dir` and `-t/--temp_dir` through to
  the pipeline (previously only `-c`).
- Env launcher is decoupled, one knob: `python_command` in the main config sets
  the command that launches python everywhere. `run_pipeline.sh` greps it from
  the config (pure bash — python is what it is resolving) and exports it as
  `TOWBINTOOLS_PYTHON` for the pre-flight and the submitted job; the workers read
  it via `get_python_command`. Exporting `TOWBINTOOLS_PYTHON` directly still wins
  (bootstrap escape hatch when the default can't even start python). Unset
  everywhere = the micromamba default, so behaviour is unchanged.
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
- Post-run cleanup script (pendant to `init_pipeline`, called by the final
  linker when there is no next block): repurpose `cleanup_temp_files.sh` into a
  configurable protocol. Candidate tasks to move there: remove the launcher's
  now-empty repo-root `sbatch_output/` landing zone (flagged in
  `setup_run_dir`), optional temp-dir clearing. Weighed against a separate
  end-of-run slurm job and rejected (queue latency) — see the log-output notes.
- Folder inputs: decouple internal references from the analysis-dir name (today
  the config repeats the prefix, so renaming `analysis_dir_name` forces
  rewriting every reference). Aim to support: (a) absolute path, (b) name-only
  (resolved under the experiment folder), (c) maybe relative. One place holds
  the dir name.
- Naming: `analysis_dir_name` / `analysis_subdir` really denote the OUTPUT
  directory. Renaming only the variables would desync them from the config key,
  and renaming the key is a breaking config change — so settle the naming as
  part of the folder-decoupling work above, not separately.

## CLI flags — the rule
- Precedence is uniform: CLI flag > config key > default. `-c/--config` is the
  only flag with no config counterpart (by definition) and is required.
- Flags cover what varies per invocation or per machine (`-e/--experiment_dir`
  where the data is, `-t/--temp_dir` where scratch goes); everything scientific
  stays in the config. Deliberately no flag for `analysis_dir_name` — it is an
  output-layout choice, and today changing it also means rewriting every
  `analysis/...` reference in the config, so a flag would be a half-measure.

## Known cleanups to mention / finish before docs
- Document custom blocks. The `CustomBuildingBlock.create_command` bug (missing
  `config` param, plus a doubled `run -n towbintools python3` launcher) is fixed;
  custom blocks now work on both backends.
- Workers use a bare `import utils` (rely on script dir on sys.path) — revisit
  when packaging.
- The outer orchestrator job's resources are now config-driven: `run_pipeline.sh`
  passes sbatch CLI flags built from `sbatch_init`, overriding the minimal header
  in `init_pipeline.sh`. The env launcher is now overridable too (see
  `TOWBINTOOLS_PYTHON` under "Which script does what").
