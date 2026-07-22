# Trade-offs taken during the refactor

Every entry is a place where a deliberate choice closed off an alternative. Kept
so the reasoning survives the commits and can be presented, revisited or
reversed later. Add to this whenever a decision has a real cost — not for
choices that are simply correct.

Format: **decision** — why — *cost* — how hard to reverse.

## Execution model

**Local backend shells out instead of running in-process.**
The local backend keeps the existing `subprocess` call and only swaps the
interpreter for `sys.executable`, rather than importing the workers and calling
them directly. Chosen for a small diff and a single code path shared with slurm.
*Cost: a process launch per block, no shared memory between blocks, and local
debugging/profiling still crosses a process boundary.*
Reversible: moderate — `run_command_local` is the only place to change.

**The block chain self-propagates through the linker.**
Each worker's job script has the linker command appended, so a block submits its
own successor. Inherited design, kept.
*Cost: no central scheduler sees the whole pipeline. There is no single process
that can report overall status, cancel the rest of a run, or notice that the
chain died — the run simply stops, and only the logs say where.*
Reversible: hard — this is the core execution model.

**The completion marker is written by the last linker, not a dedicated job.**
A separate end-of-run slurm job would sit in the queue, so the "finished" signal
could arrive long after the run really ended, and a failed submission would lose
it entirely.
*Cost: if the final job is killed (e.g. time limit) after its block finished but
before the linker ran, a run that did all its work shows no FINISHED marker.*
Reversible: easy.

## Layout and packaging

**Flat package layout, not `src/`.**
`src/` would require an install or `PYTHONPATH` for the package to be importable,
coupling every cluster invocation to a correct install. Flat keeps
`python -m towbintools_pipeline...` working from the repo root anywhere.
*Cost: the repo root is importable, so a working tree can silently shadow an
installed copy; `src/` is the modern default and avoids that class of bug.*
Reversible: moderate, but touches every invocation path.

**Workers are resolved by absolute path and use a bare `import utils`.**
They rely on their own directory being on `sys.path`. Left as-is so far.
*Cost: not package-clean; breaks if the workers are ever imported rather than
executed.*
Reversible: easy — scheduled for the packaging milestone.

**Generated job scripts, `-J` names and logs are named after the BLOCK, not the worker.**
A block can map to several workers (`segmentation` →
`learning_based_segment.py` or `non_learning_segment.py`), `sbatch_overrides` is
keyed by block name, and the progress log prints block names.
*Cost: `batch/segmentation.sh` does not match `learning_based_segment.py`, so
the filename alone does not reveal which implementation ran (the command inside
the file does).*
Reversible: easy.

**The cluster `environment.yml` still duplicates the dependency list.**
`pyproject.toml` is the single source for the local path, but the cluster env
(symlink-flip + hash-pinned pip + conda-lock) is battle-tested and was left
untouched rather than risk the working install.
*Cost: dependencies are declared twice and can drift until this is consolidated.*
Reversible: easy — planned, needs a cluster test.

## Configuration

**`sbatch_extra_options` accumulate; scalar `sbatch_*` keys replace.**
Prevents a per-block or init section from silently dropping a cluster-wide
entry such as `--account`.
*Cost: a section cannot remove a top-level option. An option that must not apply
everywhere has to be moved down into the sections instead.*
Reversible: easy.

**`sbatch_init` is a separate section, not a key inside `sbatch_overrides`.**
The outer job is not a building block and is resolved by a different program at
a different time (`run_params.py`, from bash, before the pipeline starts).
*Cost: three config sections to learn instead of two.*
Reversible: easy.

**The outer job's resources are sbatch CLI flags that override the script header.**
This is what makes a new cluster a config-only change.
*Cost: the flags are produced by a Python call before submission. If that call
fails (bad config, broken env), it silently yields nothing and the job falls
back to the bare header. Mitigated by running
`python -m towbintools_pipeline.run_params --sbatch-init -c <config>` as a
pre-flight check.*
Reversible: easy.

**Internal folder references still repeat the analysis-dir prefix.**
`analysis_dir_name` is honored everywhere, but the config still writes
`analysis/ch2_seg` in every reference.
*Cost: renaming the analysis dir means rewriting every reference in the config —
an inconsistent pair silently produces a missing-column crash mid-run.*
Reversible: moderate; deliberately deferred as config-breaking.

## Files and provenance

**Temp defaults to an in-repo `./temp_files`.**
The durable outputs and the backup are already external, and `-t <path>` covers
the large-experiment / home-quota case.
*Cost: the default writes inside the repo, which is not self-evident to a new
user.* Mitigated: gitignored, transient.
Reversible: easy.

**The pipeline reads its config from the original path; temp and backup are write-only.**
This is what makes a relative `slurm_config:` resolve against the config's own
directory — the previous behaviour copied the config into temp, ran the copy,
and silently failed to find the sibling file.
*Cost: the recorded config is a snapshot of intent taken at start-up, not proof
of what executed; blocks run from a pickled copy.*
Reversible: easy.

**`pipeline_backup/` moved beside `report/` rather than inside it.**
`report/` holds results, the backup holds provenance.
*Cost: backups from earlier runs remain in the old location; there is no
migration.*
Reversible: easy.

**The run directory and log relocation moved from bash into Python.**
Gives the temp path a single definition instead of one in bash and one in
argparse.
*Cost: if the environment is broken and Python never starts, the outer job's
logs stay in the repo-root `sbatch_output/` instead of being moved into the run
directory. Previously bash moved them first, so they landed correctly even then.*
Reversible: easy.

**Local run directories are keyed on a start timestamp with second resolution.**
Slurm uses the job id; local runs had no unique id at all and silently
overwrote each other's backup.
*Cost: two local runs started within the same second would collide. Start-up
cost makes this practically unreachable, but it is not impossible.*
Reversible: easy.

**The combined log is rebuilt at every link rather than written once at the end.**
A run that dies mid-chain still leaves a readable log.
*Cost: the file is rewritten once per block, and the tail of the block that is
running while it is written is never included.*
Reversible: easy.

**The full building-block config dump was removed from the terminal log.**
It printed the entire resolved config for every block, dwarfing everything else.
*Cost: that detail is no longer in the terminal output — it is still snapshotted
into the run backup.*
Reversible: trivial (one line).

## Process

**Comments are capped at ~2 lines and state what the code does, not why.**
Rationale lives in commit messages, `DOCS_TODO.md` and this file.
*Cost: reading the source alone does not explain why a choice was made.*
Reversible: n/a.

**Changes are stacked PRs merged bottom-up, none merged yet.**
Keeps each step reviewable in isolation.
*Cost: the stack is long; a change low in it means rebasing everything above.*
Reversible: n/a.

**`gui/`, `training/` and `tools/` were left untouched.**
Scope control — the core pipeline first.
*Cost: they may not work against the new layout until adapted.*
Reversible: n/a — scheduled.
