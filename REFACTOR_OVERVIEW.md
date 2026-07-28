# Towbintools Pipeline — Refactor Overview

A guide to what changed in the pipeline and **why**, for presenting the work and
for onboarding. It is organised by theme, not by the order things were built.

For the *cost* of each decision see [`TRADEOFFS.md`](TRADEOFFS.md); for the
running notes that feed the eventual user-docs rewrite see
[`DOCS_TODO.md`](DOCS_TODO.md). This file is the high-level map that ties them
together. Links to the pull request behind each theme are collected at the
[end](#where-to-find-each-change).

## At a glance

The pipeline was a cluster-only collection of scripts: it ran only as SLURM jobs,
had to be launched from a specific place with the right paths, and cluster- and
environment-specific details were baked into the code. Adapting it, testing it,
or even running it off the cluster was hard.

It is now an **installable Python package** that runs **locally or on SLURM**
through one code path, **adapts to a new cluster by editing config alone**,
**validates its config up front**, and is **covered by automated tests that run
on every change**. The goal throughout was maintainability: make it easy to run,
easy to change safely, and easy to reason about.

## Nothing breaks — the old ways still work

None of this is a hard cut-over. Existing use keeps working:

- **Existing configs still run** — folder references may still be written with
  the old prefix, single-file configs still work, and the new validation only
  rejects genuinely broken configs.
- **The pipeline is still launched the same way** on the cluster
  (`bash scripts/run_pipeline.sh ...`), and running it directly with
  `python -m towbintools_pipeline.init_pipeline ...` still works — the new
  installed `towbintools-pipeline` command is an *addition*, not a replacement.
- **Every new behaviour that could change a run is opt-in and defaults to the
  old behaviour** — the backend defaults to SLURM, an unset launcher means the
  previous micromamba command, and the end-of-run cleanup defaults to off.

## What changed, by theme

### 1. Run it anywhere — a local backend
The pipeline can now run entirely on one machine (`backend: local`) through the
same code path it uses on the cluster, instead of only as SLURM jobs.

**Why —** you can develop, test, debug, and demo without a cluster, and this is
the foundation that made automated testing possible at all.

### 2. A real, installable package
Installing the project from its checkout (`pip install`) now registers a
`towbintools-pipeline` command that runs from any directory; each analysis step
runs as a proper Python module; and the default config and model ship inside the
package. (It is installed from the repository, not from a public index like PyPI
— publishing there would be a later step if ever wanted.)

**Why —** reproducible installs, no path/`PYTHONPATH` juggling, and something
that can actually be distributed and onboarded.

### 3. Not tied to one environment manager
Micromamba and a specific environment name used to be hardcoded in the code and
scripts. Now a single `python_command` config key decides how Python is launched
everywhere (with an environment-variable escape hatch), and there is a separate
local installer that needs no micromamba.

**Why —** teams can use conda, venv, micromamba — whatever they have — and new
users get a much simpler setup.

### 4. Adapt to a new cluster by editing config, not code
SLURM resources now live in a dedicated `slurm_config.yaml` with shared defaults,
per-analysis-step overrides, and the orchestrator job's own resources, plus a
free-form list for cluster-specific flags (e.g. `--account`, `--mem-per-cpu`,
partitions) that previously required editing the code.

**Why —** moving to a different cluster, or changing account/partition/memory, is
a config-only change — no edits to the pipeline code.

### 5. Simpler, less error-prone configuration
Folder references used to repeat the analysis-directory prefix everywhere, and
renaming that directory meant rewriting the config. References are now written by
name, resolved automatically, and survive renaming the output directory; the
shipped config carries commented examples for the main knobs.

**Why —** less repetition and fewer foot-guns when writing a config.

### 6. Fail fast on a bad config
A mistake in the config used to surface late — mid-run, or as a confusing error
deep inside a job. The config is now validated up front, and the run stops before
anything is created, reporting **all** the problems at once.

**Why —** mistakes are caught in seconds with clear messages, and there are no
half-started runs to clean up.

### 7. Run isolation and provenance
Each run now gets its own working directory, so repeated runs can no longer
overwrite one another (previously local runs did). A snapshot of the exact config
and the code version is recorded for **every** run — not just on the cluster —
and the run's backup sits beside the results as provenance. An opt-in setting
reclaims scratch space after a successful run. (The durable outputs were already
written outside the repository; that did not change.)

**Why —** reproducibility, no accidental overwrites, and a clear record of what
actually ran.

### 8. Observability — know what ran and where it stopped
The run prints its planned sequence of steps and a start/finish marker around
each one, and keeps a single combined log that is rebuilt as the run progresses,
ending with a clear "finished" vs "still running" marker.

**Why —** you can watch progress and diagnose a stalled run from the logs alone.

### 9. A repository that explains itself
The layout is now four clear tiers — the **package** (the pipeline itself), the
**deployment glue** (environment definitions + operational scripts), the
**extras** (tools, GUI, training), and the **meta/docs** — with scripts grouped
by what they do (set up / run / maintain). The old "auto-reset your checkout from
git on every launch" behaviour was replaced by an explicit update command.

**Why —** a newcomer can tell what each part is for, and launching a run no longer
silently changes your working copy.

### 10. A safety net — tests and continuous integration
There is now an end-to-end smoke test plus unit tests for the core logic, run
automatically on every push and pull request (a green check gates changes), and a
consistent code style across the package.

**Why —** the refactor itself was validated continuously, and future changes are
protected from silent regressions.

### How the work was done
Changes were made as small, reviewable, stacked pull requests rather than one
large rewrite, each staying close to the original code where no change was
needed, and every decision that carried a real cost was written down (see
`TRADEOFFS.md`).

**Why —** the work stays auditable, and the reasoning survives beyond the commits.

## Status and what's next

- **Verified on the cluster** through packaging, the deployment-layout change, and
  the cluster package install. Config validation and CI are green.
- **Pending a confirmation cluster run** (low risk): the opt-in end-of-run cleanup
  and the folder-reference change — both tested locally.
- **Deliberately left for later:**
  - the **extras** (`tools/`, `gui/`, `training/`) are not yet adapted to the new
    layout and conventions;
  - the **user-facing documentation** (README + book) rewrite is the final step,
    to be done *after* this discussion, driven from `DOCS_TODO.md`.

## Where to find each change

Each theme corresponds to one (occasionally a few) stacked pull request on the
fork, reviewed and merged bottom-up. The links below open the diff for that PR
(`base…target` compare view) on `github.com/quasar1357/towbintools_pipeline`.

1. **Local backend** — [`main…feature/local-backend`](https://github.com/quasar1357/towbintools_pipeline/compare/main...feature/local-backend)
2. **Installable package** — [`…feature/packaging-entry-point`](https://github.com/quasar1357/towbintools_pipeline/compare/chore/retire-git-self-update...feature/packaging-entry-point) (then `refactor/deployment-layout`, `feature/cluster-package-install`)
3. **Env-manager decoupling** — [`…feature/launcher-decoupling`](https://github.com/quasar1357/towbintools_pipeline/compare/refactor/script-responsibilities...feature/launcher-decoupling)
4. **Cluster-adapt-by-config** — [`…feature/slurm-config`](https://github.com/quasar1357/towbintools_pipeline/compare/feature/env-install...feature/slurm-config) (then `feature/slurm-per-block`, `feature/slurm-outer-script`)
5. **Config & folder references** — [`…feature/folder-ref-decoupling`](https://github.com/quasar1357/towbintools_pipeline/compare/feature/cleanup-on-success...feature/folder-ref-decoupling)
6. **Fail-fast validation** — [`…feature/config-validation-ci`](https://github.com/quasar1357/towbintools_pipeline/compare/feature/folder-ref-decoupling...feature/config-validation-ci) (commit `821f9c4`)
7. **Run isolation & provenance** — [`…feature/externalize-io`](https://github.com/quasar1357/towbintools_pipeline/compare/feature/slurm-config...feature/externalize-io) (then `fix/config-loading-and-backup`, `feature/cleanup-on-success`)
8. **Observability & logging** — [`…feature/block-progress-logging`](https://github.com/quasar1357/towbintools_pipeline/compare/chore/cleanup-and-temp-default...feature/block-progress-logging)
9. **Repo layout & deployment** — [`…refactor/repo-structure`](https://github.com/quasar1357/towbintools_pipeline/compare/feature/externalize-io...refactor/repo-structure) (then `refactor/deployment-layout`, `chore/retire-git-self-update`)
10. **Tests & CI** — same PR as validation, [`…feature/config-validation-ci`](https://github.com/quasar1357/towbintools_pipeline/compare/feature/folder-ref-decoupling...feature/config-validation-ci) (commits `def211d`, `3cec1db`)

## The three documents, going forward

- **`REFACTOR_OVERVIEW.md`** (this file) — the narrative: what changed and why.
  Use it to present the work and to onboard someone new. Update it when a whole
  new theme lands.
- **`TRADEOFFS.md`** — the decision ledger: every change that cost something,
  split into *introduced by this refactor* (the ones to scrutinise) and
  *inherited and consciously kept*. Reach for it when someone asks "why did you
  do X, and what did it cost?"
- **`DOCS_TODO.md`** — the working feedstock for the eventual user-docs rewrite,
  plus deferred engineering TODOs and the agreed code conventions. Not a
  presentation document; it drives the docs step and the non-core follow-up.
