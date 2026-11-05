"""The observation page draws overview_charts.json specs at the overview's markers."""

import json
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "policy_finder" / "observation_app"


@pytest.fixture
def export_html(tmp_path, monkeypatch):
    monkeypatch.setenv("PF_WORKTREE_ROOT", str(tmp_path))
    monkeypatch.syspath_prepend(str(APP))
    for name in ("export_html", "dashboard"):
        sys.modules.pop(name, None)
    import export_html

    yield export_html
    for name in ("export_html", "dashboard"):
        sys.modules.pop(name, None)


def test_markers_become_charts_and_unknown_ones_vanish(export_html):
    charts = export_html.Charts()
    text = "<p>a</p><!--chart:gap--><p>b</p><!--chart:missing-->"
    out = charts.place_markers(text, {"gap": {"mark": "line"}})
    assert out == '<p>a</p><div class="chart" data-chart="0"></div><p>b</p>'
    assert charts.specs == [{"mark": "line", "width": "container"}]


def test_overview_charts_reads_the_file_beside_the_overview(export_html, tmp_path):
    d = export_html.d
    assert d.overview_charts() == {}
    (tmp_path / "overview_charts.json").write_text(json.dumps({"gap": {"mark": "bar"}}))
    assert d.overview_charts() == {"gap": {"mark": "bar"}}
