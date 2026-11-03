"""CLI accepts orchestrator YAMLs (params nested) and flat configs."""

import textwrap
from pathlib import Path

import pytest
import yaml

from aimanager.cli import load_and_validate


def _write(tmp_path, body):
    path = tmp_path / "config.yml"
    path.write_text(textwrap.dedent(body))
    return str(path)


def test_unwraps_orchestrator_params(tmp_path):
    path = _write(
        tmp_path,
        """
        exec:
          cores: 4
          memory: 8
        params:
          data_file: experiments/x.csv
          model_args: {}
          train_args: {}
          optimizer_args: {}
        """,
    )
    config = load_and_validate(path, "train-ah")
    assert config["data_file"] == "experiments/x.csv"
    assert "exec" not in config


def test_flat_config_is_unchanged(tmp_path):
    path = _write(
        tmp_path,
        """
        data_file: experiments/x.csv
        model_args: {}
        train_args: {epochs: 1}
        optimizer_args: {}
        """,
    )
    config = load_and_validate(path, "train-ah")
    assert config["train_args"] == {"epochs": 1}


def test_grid_is_left_to_the_ssh_orchestrator(tmp_path):
    path = _write(
        tmp_path,
        """
        grid: {}
        params:
          data_file: experiments/x.csv
        """,
    )
    with pytest.raises(SystemExit):
        load_and_validate(path, "train-ah")


def test_job_id_from_params_labels(tmp_path):
    path = _write(
        tmp_path,
        """
        params:
          data_file: experiments/x.csv
          model_args: {}
          train_args: {}
          optimizer_args: {}
          labels:
            architecture: mlp+rnn+edge
            dataset: 50ep_doubled
        """,
    )
    config = load_and_validate(path, "train-ah")
    assert config["job_id"] == "architecture_mlp+rnn+edge__dataset_50ep_doubled"


def test_job_id_prefers_the_jobs_own_labels(tmp_path):
    path = _write(
        tmp_path,
        """
        labels: {run: a}
        params:
          data_file: experiments/x.csv
          model_args: {}
          train_args: {}
          optimizer_args: {}
          labels: {run: b}
        """,
    )
    assert load_and_validate(path, "train-ah")["job_id"] == "run_a"


def test_job_id_is_kept_and_falls_back_to_the_file_name(tmp_path):
    flat = """
        data_file: experiments/x.csv
        model_args: {}
        train_args: {}
        optimizer_args: {}
        """
    path = _write(tmp_path, flat + "job_id: mine\n")
    assert load_and_validate(path, "train-ah")["job_id"] == "mine"
    path = _write(tmp_path, flat)
    assert load_and_validate(path, "train-ah")["job_id"] == "config"


def test_simulate_config_gets_no_job_id(tmp_path):
    path = _write(
        tmp_path,
        """
        artificial_humans: {}
        managers: {}
        n_episodes: 1
        n_episode_steps: 1
        """,
    )
    assert "job_id" not in load_and_validate(path, "simulate")


@pytest.mark.parametrize(
    "config",
    sorted(
        str(p)
        for p in Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("configs/training/artificial_humans")
        .rglob("*.yml")
    ),
)
def test_repo_ah_configs_get_the_orchestrators_job_id(config):
    """Every repo AH config the CLI accepts gets run.py's job id."""
    with open(config) as f:
        raw = yaml.safe_load(f)
    if "grid" in raw or "include" in raw:
        pytest.skip("grid/include configs stay with the SSH orchestrator")
    labels = raw.get("labels", raw.get("params", {}).get("labels", {}))
    expected = "__".join(f"{k}_{v}" for k, v in labels.items())
    assert load_and_validate(config, "train-ah")["job_id"] == expected
