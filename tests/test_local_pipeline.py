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


def _write_image(path, blob_width):
    # 3-channel image with a bright rectangle in channel 0 (a worm proxy).
    image = np.zeros((3, 128, 128), dtype=np.uint16)
    image[0, 30:100, 20 : 20 + blob_width] = 4000
    # Store channels as separate planes (explicit, so the reader sees 3 channels).
    tifffile.imwrite(str(path), image, photometric="minisblack")


def _build_experiment(tmp_path, config_experiment_dir=None):
    # Tiny 2-image raw experiment + a local-backend config; returns config_path.
    # config_experiment_dir overrides what the config records as experiment_dir.
    raw_dir = tmp_path / "exp" / "raw"
    raw_dir.mkdir(parents=True)
    _write_image(raw_dir / "Time000000_Point000000_synthetic.tiff", 50)
    _write_image(raw_dir / "Time000001_Point000000_synthetic.tiff", 55)

    config = {
        "experiment_dir": config_experiment_dir or str(tmp_path / "exp"),
        "analysis_dir_name": "analysis",
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
        "morphology_computation_masks": ["analysis/ch1_seg"],
        "morphological_features": [["area"]],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


def _run_pipeline(config_path, extra_args=()):
    # Run from the repo root so `-m towbintools_pipeline...` resolves; the local
    # backend launches the workers with this same interpreter (sys.executable).
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
        cwd=REPO_ROOT,
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


def test_local_pipeline_default_temp_dir_next_to_experiment(tmp_path):
    # Without --temp_dir (and no temp_dir config key), temp files default next
    # to the experiment data, not the repo/cwd.
    config_path = _build_experiment(tmp_path)

    result = _run_pipeline(config_path)
    assert (
        result.returncode == 0
    ), f"pipeline failed:\n{result.stdout}\n{result.stderr}"

    default_temp = tmp_path / "exp" / "temp_files"
    assert (default_temp / "pickles").is_dir()
    # the run still produced its outputs under the experiment dir
    morph_csv = tmp_path / "exp" / "analysis" / "report" / "ch1_seg_morphology.csv"
    assert morph_csv.exists()


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
