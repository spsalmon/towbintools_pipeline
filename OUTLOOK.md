# Outlook — future work & deferred engineering

Forward-looking notes: what is still to come, and the engineering/design cleanups
consciously deferred. The high-level "what's next" lives in
[`REFACTOR_OVERVIEW.md`](REFACTOR_OVERVIEW.md) — its three major follow-ups are the
top three below. Decisions that cost something are in
[`TRADEOFFS.md`](TRADEOFFS.md); the user-docs feedstock is in
[`DOCS_TODO.md`](DOCS_TODO.md). Section names in quotes below refer to DOCS_TODO
sections.

## Future work (roughly priority-ordered)

Single index of what is still to come; details live in DOCS_TODO where noted.
Higher items first; the top three are the overview's major follow-ups, the rest
are smaller. (DONE: config-validation of input paths + unknown-key rejection —
see "Config validation"; the `init-configs` scaffolding command and the
subcommand dispatcher — see "CLI / commands".)

1. **Extras adaptation (PR F)** — bring `tools/`, `gui/`, `training/`,
   `examples/custom_scripts/` onto the new layout + code conventions (see the
   "Code conventions" scope note). Deferred until the core path is agreed.
2. **Docs rewrite (PR G, last)** — README + `book/` overhaul, driven from
   DOCS_TODO. Only after the core overhaul is agreed with the product owner.
3. **"Real" API** — an object-oriented / stepwise outer orchestration in Python
   (drive blocks one at a time, opt-in linking) alongside the current config-driven
   run. Larger design effort; optional.
4. **Publish to PyPI** — currently installed from the repo/checkout only; publishing
   would make `pip install towbintools-pipeline` work directly. Easy later step.
5. **Config validation, further** — optional warning-level "contents reasonable"
   checks (e.g. `experiment_dir` contains a `raw/`), and running `validate_config`
   in the login-node pre-flight (see the follow-up under "Config validation").
6. **(lower) Warning-log volume** — some warnings can fire once per image and blow
   up the `.out` file(s). Idea: keep a list of ignorable warnings in a repo file
   (adjustable, but out of the user's config surface) and filter those when logging.
7. **(lower) Output-filename suffix** — optional (bool, default off) suffix on output
   names. Deferred/parked: the data handling / read-in is changing soon, so not worth
   doing against the current scheme.

## Deferred engineering cleanup

- Cluster dep de-duplication was CONSIDERED and dropped: the cluster build is a
  hash-pinned `--require-hashes --no-deps` install from `conda-linux-64.lock`, so
  a `- .` (unhashable local path) does not fit, and `environment.yml` is a
  superset of pyproject (gui/training/cellpose/bioio, conda-delivered). Instead
  the package is installed separately, editable + `--no-deps`, by
  `install_pipeline.sh`/`update_pipeline.sh` after the env build — lock untouched.
  A future pyproject-extras approach (gui/training groups) could revisit de-dup.

## Deferred design / cleanup (later PRs)

- Folder inputs, further: `resolve_ref` covers name-only refs (done). Still open
  if needed: first-class (a) absolute and (c) relative external-directory refs
  (today an absolute path passes through, but there is no relative-to-experiment
  form) — add only if cross-experiment refs are actually wanted.
- Naming: `analysis_dir_name` / `analysis_subdir` really denote the OUTPUT
  directory. Renaming the KEY is a breaking config change, so it stays deferred
  and separate from the (now-done) ref decoupling.

## Robustness findings from a local end-to-end run (found 2026-08-25)

Surfaced while validating the fork with a local pip install + smoke test. None
block the merge (several confirmed pre-existing in the original); all are
robustness / UX / packaging improvements. Handling principle: prefer fixing the
CODE to be version-robust over pinning; reserve version constraints for genuinely
external API breaks, and use loose caps — never an exact pin just because a newer
version exists.

1. **no_timepoints + get_experiment_time crash (PRE-EXISTING).** When raw
   filenames don't match `time_regex`/`point_regex`, the filemap falls back to a
   no-timepoints frame (`config["no_timepoints"]=True`), but the guard in
   `init_pipeline.py` checks only `get_experiment_time`, not `no_timepoints`, so
   it still calls `get_experiment_time_from_filemap`, which needs a `Time` column
   → `ColumnNotFoundError`. Fix: skip experiment-time extraction (or return null
   ExperimentTime) when `no_timepoints`.
2. **Poisoned filemap cache.** The filemap is regenerated only if absent, but it
   is written before the experiment-time step can crash — so a mid-init crash
   leaves a stale/broken filemap that survives config/regex fixes until the
   analysis output dir is deleted. Fix: don't persist until valid, or invalidate
   on crash.
3. **Raw column named after the folder; block refs hardcode `"raw"`.** The raw
   column is `raw_dir_name`; `resolve_ref` passes a ref through only if it equals
   `raw_dir_name` (or is absolute), else prefixes `analysis_dir_name/`. So when
   `raw_dir_name != "raw"`, every `"raw"` block ref breaks. Fix: treat literal
   `"raw"` as always the raw input, regardless of `raw_dir_name`.
4. **cellpose imported at module scope.** `workers/segmentation_learning_based.py`
   does `from cellpose import models` at import time, so the module fails without
   cellpose even for `deep_learning`. Fix: lazy-import cellpose inside the cellpose
   path; same idea for other heavy method-specific imports.
5. **pyproject under-declares deps (overlaps the PyPI item above).** The pipeline
   directly imports cv2/scipy/torch/xgboost/pandas/tqdm but declares none — they
   arrive transitively via towbintools, so the set is fragile (an unpinned
   cellpose install pulled `opencv-python-headless` over the `opencv-contrib`
   towbintools needs → straightening crashed). Fix: declare the direct imports;
   put heavy backends (cellpose, segment_anything) behind extras; add loose caps
   for known-bad combos (e.g. opencv-contrib 4.x).
6. **quality_control crashes when all samples are eggs.** If the egg
   pre-classifier labels every sample egg, `non_egg_indices` is empty → empty
   xgboost DMatrix → `np.argmax(..., axis=1)` raises. Whether it crashes is
   xgboost-version-dependent (3.4.1 raises, 3.3.0 tolerated). Fix
   (version-independent): short-circuit when there are no non-egg samples.
