"""Tests for the evaluation-suite main script (#128)."""

from pathlib import Path

import pytest

from aimanager.evaluation_suite.convert import (
    HUMAN_DATA_FILE,
    SIM_EXAMPLE_FILE,
    load_human,
    load_sim,
)
from aimanager.evaluation_suite.evaluate import evaluate, run_cli
from aimanager.evaluation_suite.metrics import GROUPS

REPO = Path(__file__).resolve().parents[3]
N_ROWS = sum(len(g.KINDS) for g in GROUPS.values())


def test_evaluate_produces_one_row_per_pairing_and_metric():
    human = load_human(REPO / HUMAN_DATA_FILE)
    sims = load_sim(REPO / SIM_EXAMPLE_FILE)
    result = evaluate(human, sims)

    assert len(result) == N_ROWS * len(sims)
    assert (result["d"] >= 0).all()
    assert result["d"].notna().all()
    # diagnostics only on CA/CC/CE
    assert set(result.loc[result["std_diff"].notna(), "metric"]) == {
        "CA",
        "CC",
        "CE",
    }


def test_run_cli_fails_clearly_without_parquet(tmp_path, capsys):
    config = {"output_dir": str(tmp_path / "nowhere")}
    with pytest.raises(SystemExit):
        run_cli(config, "some_config.yml")
    assert "save_per_round" in capsys.readouterr().err


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_run_cli_writes_metrics_and_scores(tmp_path):
    # point a minimal config at the real per_round.parquet via a symlink
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "per_round.parquet").symlink_to(REPO / SIM_EXAMPLE_FILE)
    run_cli({"output_dir": str(output_dir), "scoring_repeats": 2}, "config.yml")
    metrics = output_dir / "evaluation" / "metrics.csv"
    assert metrics.exists()
    assert sum(1 for _ in open(metrics)) == 3 * N_ROWS + 1  # pairings + header
    scores = output_dir / "evaluation" / "scores.csv"
    assert scores.exists()
    assert sum(1 for _ in open(scores)) == 3 * N_ROWS + 1
