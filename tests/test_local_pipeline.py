"""End-to-end smoke test for the local execution backend.

Generates tiny synthetic images, runs the pipeline with `backend: local`
(threshold segmentation -> area morphology), and checks the outputs. No
slurm, no micromamba, no bundled data.
"""
import csv
import os
import subprocess
import sys

import pytest

pytest.importorskip("towbintools")
np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
yaml = pytest.importorskip("yaml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # import the package directly for unit tests


def _write_image(path, blob_width):
    # 3-channel image with a bright rectangle in channel 0 (a worm proxy).
    image = np.zeros((3, 128, 128), dtype=np.uint16)
    image[0, 30:100, 20 : 20 + blob_width] = 4000
    # Store channels as separate planes (explicit, so the reader sees 3 channels).
    tifffile.imwrite(str(path), image, photometric="minisblack")


def _build_experiment(tmp_path, config_experiment_dir=None, analysis_dir_name="analysis"):
    # Tiny 2-image raw experiment + a local-backend config; returns config_path.
    # config_experiment_dir overrides what the config records as experiment_dir.
    raw_dir = tmp_path / "exp" / "raw"
    raw_dir.mkdir(parents=True)
    _write_image(raw_dir / "Time000000_Point000000_synthetic.tiff", 50)
    _write_image(raw_dir / "Time000001_Point000000_synthetic.tiff", 55)

    config = {
        "experiment_dir": config_experiment_dir or str(tmp_path / "exp"),
        "analysis_dir_name": analysis_dir_name,
        "raw_dir_name": "raw",
        "report_format": "csv",
        "pixelsize": [0.65],
        "backend": "local",
        "get_experiment_time": False,
        "n_jobs": 1,
        "building_blocks": ["segmentation", "morphology_computation"],
        "segmentation_column": ["raw"],
        "segmentation_method": ["threshold"],
        "segmentation_channels": [[0]],
        "segmentation_name_suffix": [None],
        "morphology_computation_masks": [f"{analysis_dir_name}/ch1_seg"],
        "morphological_features": [["area"]],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


def _run_pipeline(config_path, extra_args=(), cwd=REPO_ROOT):
    # PYTHONPATH lets `-m towbintools_pipeline...` resolve from any cwd; the
    # local backend launches the workers with this same interpreter and resolves
    # their scripts by absolute path, so cwd only affects a relative temp dir.
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "towbintools_pipeline.init_pipeline",
            "-c",
            str(config_path),
            *extra_args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_local_pipeline_segmentation_and_morphology(tmp_path):
    config_path = _build_experiment(tmp_path)
    temp_dir = tmp_path / "pipeline_temp"

    result = _run_pipeline(config_path, ["--temp_dir", str(temp_dir)])
    assert (
        result.returncode == 0
    ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    # segmentation produced one mask per input image
    mask_dir = tmp_path / "exp" / "analysis" / "ch1_seg"
    masks = sorted(mask_dir.glob("*.tiff"))
    assert len(masks) == 2

    # morphology produced one record per image with a positive area
    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()
    with open(morph_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(float(r["ch1_seg_area"]) > 0 for r in rows)


def test_local_pipeline_default_temp_dir_in_cwd(tmp_path):
    # Without --temp_dir (and no temp_dir config key), temp files default to
    # ./temp_files in the working directory (matching the sbatch launcher).
    config_path = _build_experiment(tmp_path)

    result = _run_pipeline(config_path, cwd=str(tmp_path))
    assert (
        result.returncode == 0
    ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    assert (tmp_path / "temp_files" / "pickles").is_dir()
    # the run still produced its outputs under the experiment dir
    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()


def test_block_progress_is_logged(tmp_path):
    # Every block announces its start and its completion, numbered within the
    # planned sequence, and the run ends with the closing line.
    config_path = _build_experiment(tmp_path)

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert (
        result.returncode == 0
    ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    markers = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("###") or line.startswith("  1/") or line.startswith("  2/")
    ]
    assert markers == [
        "### Initializing the pipeline for subdir None ###",
        "### Pipeline plan: 2 blocks ###",
        "1/2 segmentation",
        "2/2 morphology_computation",
        "### Starting block 1/2 segmentation ###",
        "### Finished block 1/2 segmentation ###",
        "### Starting block 2/2 morphology_computation ###",
        "### Finished block 2/2 morphology_computation ###",
        "### End of the pipeline! (2 blocks) ###",
    ]


def test_concatenate_sbatch_logs(tmp_path):
    # Per-block logs are joined per stream in write order, headed by their file
    # name, without touching the originals.
    import os as _os

    from towbintools_pipeline.utils import concatenate_sbatch_logs

    temp_dir = tmp_path / "pipeline_4242"
    log_dir = temp_dir / "sbatch_output"
    log_dir.mkdir(parents=True)

    # Written out of order on purpose; mtimes decide the order, not the names.
    (log_dir / "straightening-3.out").write_text("third\n")
    (log_dir / "init-1.out").write_text("first\n")
    (log_dir / "segmentation-2.out").write_text("second\n")
    (log_dir / "segmentation-2.err").write_text("a warning\n")
    for name, stamp in (("init-1.out", 1000), ("segmentation-2.out", 2000),
                        ("straightening-3.out", 3000)):
        _os.utime(log_dir / name, (stamp, stamp))

    concatenate_sbatch_logs(str(temp_dir))

    combined = (temp_dir / "pipeline-4242.out").read_text()
    assert combined.index("first") < combined.index("second") < combined.index("third")
    assert "===== init-1.out =====" in combined
    # each stream gets its own file, and the originals stay put
    assert (temp_dir / "pipeline-4242.err").read_text().count("a warning") == 1
    assert (log_dir / "init-1.out").read_text() == "first\n"

    # Rebuilding is idempotent: the combined file lives outside the log dir, so
    # it is never folded into itself.
    concatenate_sbatch_logs(str(temp_dir))
    assert (temp_dir / "pipeline-4242.out").read_text() == combined


def test_concatenate_sbatch_logs_without_logs(tmp_path):
    # A local run has no sbatch logs at all; that must be a quiet no-op.
    from towbintools_pipeline.utils import concatenate_sbatch_logs

    concatenate_sbatch_logs(str(tmp_path))  # no sbatch_output dir
    (tmp_path / "sbatch_output").mkdir()
    concatenate_sbatch_logs(str(tmp_path))  # empty sbatch_output dir
    assert list(tmp_path.glob("pipeline-*")) == []


def test_run_config_and_version_info_backed_up(tmp_path):
    # The config and a git_info.txt land in the backup, which sits beside the
    # report (under the analysis dir), not inside it.
    config_path = _build_experiment(tmp_path)

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert (
        result.returncode == 0
    ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    backup = tmp_path / "exp" / "analysis" / "pipeline_backup" / "pipeline_temp"
    assert (backup / "config.yaml").exists()
    assert (backup / "git_info.txt").exists()


def test_custom_analysis_dir_name(tmp_path):
    # A non-default analysis_dir_name flows through: outputs land under it and
    # the downstream morphology block resolves its mask input correctly.
    config_path = _build_experiment(tmp_path, analysis_dir_name="results")

    result = _run_pipeline(config_path, ["--temp_dir", str(tmp_path / "pipeline_temp")])
    assert (
        result.returncode == 0
    ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    masks = sorted((tmp_path / "exp" / "results" / "ch1_seg").glob("*.tiff"))
    assert len(masks) == 2
    morph_csv = tmp_path / "exp" / "results" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()
    with open(morph_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(float(r["ch1_seg_area"]) > 0 for r in rows)


def test_merge_slurm_config_resolves_relative_sibling(tmp_path):
    # A relative slurm_config resolves next to the main config file; its keys
    # merge in, but inline sbatch_* keys still win.
    from towbintools_pipeline.utils import merge_slurm_config

    (tmp_path / "slurm_config.yaml").write_text(
        "sbatch_time: 0-02:00:00\nsbatch_cpus: 8\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "backend: slurm\nslurm_config: slurm_config.yaml\nsbatch_cpus: 4\n"
    )

    config = yaml.safe_load(config_path.read_text())
    merged = merge_slurm_config(config, str(config_path))
    assert merged["sbatch_time"] == "0-02:00:00"  # from the file
    assert merged["sbatch_cpus"] == 4  # inline wins


def test_merge_slurm_config_missing_is_skipped(tmp_path):
    # A missing slurm_config is skipped, leaving the config untouched.
    from towbintools_pipeline.utils import merge_slurm_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text("backend: slurm\nslurm_config: does_not_exist.yaml\n")

    config = yaml.safe_load(config_path.read_text())
    merged = merge_slurm_config(config, str(config_path))
    assert "sbatch_time" not in merged


def test_resolve_block_slurm_default_and_override():
    # A block with no override gets the shared defaults; an overridden type gets
    # the defaults merged with its entry (the override winning per key).
    from towbintools_pipeline.utils import resolve_block_slurm

    config = {
        "sbatch_cpus": 8,
        "sbatch_memory": "16G",
        "sbatch_time": "0-02:00:00",
        "sbatch_overrides": {
            "segmentation": {"sbatch_gpus": "rtx6000:1", "sbatch_memory": "32G"}
        },
    }

    morph = resolve_block_slurm(config, "morphology_computation")
    assert morph == {"sbatch_cpus": 8, "sbatch_memory": "16G", "sbatch_time": "0-02:00:00"}

    seg = resolve_block_slurm(config, "segmentation")
    assert seg["sbatch_gpus"] == "rtx6000:1"
    assert seg["sbatch_memory"] == "32G"  # override wins
    assert seg["sbatch_cpus"] == 8  # default kept


def test_resolve_init_slurm():
    # The outer job gets the defaults overlaid with sbatch_init.
    from towbintools_pipeline.utils import resolve_init_slurm

    config = {
        "sbatch_cpus": 32,
        "sbatch_memory": "64G",
        "sbatch_init": {"sbatch_cpus": 4, "sbatch_memory": "8G"},
    }

    init = resolve_init_slurm(config)
    assert init == {"sbatch_cpus": 4, "sbatch_memory": "8G"}


def test_resolve_init_slurm_drops_gpu():
    # The orchestrator job never needs the GPU, so it isn't inherited from the
    # shared defaults (but shared extras like --account still flow through).
    from towbintools_pipeline.utils import resolve_init_slurm

    config = {
        "sbatch_cpus": 32,
        "sbatch_gpus": "rtx6000:1",
        "sbatch_extra_options": ["--account=gratis"],
        "sbatch_init": {"sbatch_cpus": 4},
    }

    init = resolve_init_slurm(config)
    assert "sbatch_gpus" not in init
    assert init["sbatch_cpus"] == 4
    assert init["sbatch_extra_options"] == ["--account=gratis"]


def test_slurm_extra_options_accumulate():
    # Scalar keys are replaced by a section, but sbatch_extra_options entries are
    # appended, so a cluster-wide option is never silently dropped.
    from towbintools_pipeline.utils import resolve_block_slurm
    from towbintools_pipeline.utils import resolve_init_slurm

    config = {
        "sbatch_cpus": 32,
        "sbatch_extra_options": ["--account=gratis"],
        "sbatch_overrides": {
            "segmentation": {
                "sbatch_cpus": 8,
                "sbatch_extra_options": ["--partition=gpu"],
            }
        },
        "sbatch_init": {"sbatch_extra_options": ["--gres=pipelinecapacity:1"]},
    }

    seg = resolve_block_slurm(config, "segmentation")
    assert seg["sbatch_cpus"] == 8  # scalar replaced
    assert seg["sbatch_extra_options"] == ["--account=gratis", "--partition=gpu"]

    # A block without an override keeps only the shared extras.
    other = resolve_block_slurm(config, "straightening")
    assert other["sbatch_extra_options"] == ["--account=gratis"]

    init = resolve_init_slurm(config)
    assert init["sbatch_extra_options"] == [
        "--account=gratis",
        "--gres=pipelinecapacity:1",
    ]


def test_build_resource_directives():
    # Standard directives emitted only when set; extras appended verbatim.
    from towbintools_pipeline.utils import build_resource_directives

    assert build_resource_directives(8, "0-02:00:00", "16G", "rtx6000:1", None) == [
        "-c 8",
        "-t 0-02:00:00",
        "--mem=16G",
        "--gres=gpu:rtx6000:1",
    ]
    # memory omitted, replaced via an extra option
    assert build_resource_directives(4, None, None, None, ["--mem-per-cpu=4G"]) == [
        "-c 4",
        "--mem-per-cpu=4G",
    ]


def test_run_params_sbatch_init_flags():
    # The helper turns sbatch_init (over the defaults, minus GPU) into flags.
    from towbintools_pipeline.run_params import sbatch_init_flags

    config = {
        "sbatch_cpus": 32,
        "sbatch_memory": "64G",
        "sbatch_gpus": "rtx6000:1",
        "sbatch_extra_options": ["--account=gratis"],
        "sbatch_init": {"sbatch_cpus": 4, "sbatch_memory": "8G", "sbatch_time": "0-12:00:00"},
    }

    flags = sbatch_init_flags(config)
    assert flags == ["-c 4", "-t 0-12:00:00", "--mem=8G", "--account=gratis"]


def test_experiment_dir_cli_overrides_config(tmp_path):
    # The config records a wrong experiment_dir; --experiment_dir points at the
    # real data and must win, so the run finds the images and produces outputs.
    config_path = _build_experiment(
        tmp_path, config_experiment_dir=str(tmp_path / "wrong")
    )

    result = _run_pipeline(
        config_path,
        [
            "--experiment_dir",
            str(tmp_path / "exp"),
            "--temp_dir",
            str(tmp_path / "pipeline_temp"),
        ],
    )
    assert (
        result.returncode == 0
    ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()
