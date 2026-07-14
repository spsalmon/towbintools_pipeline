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
- (Once packaged) `pip install .` / `pip install -e .` via pyproject.toml, and
  how that relates to the conda/lock install.

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
- Document `backend`, and (once split out) where SLURM resources live and how to
  set them per block.

## Known cleanups to mention / finish before docs
- `CustomBuildingBlock.create_command` was broken (missing `config` param) — fix
  and document custom blocks.
- Workers use a bare `import utils` (rely on script dir on sys.path) — revisit
  when packaging.
