from __future__ import annotations

from solarvision import cli


def test_demo_runs(capsys):
    cli.demo()
    out = capsys.readouterr().out
    assert "SolarVision demo" in out
    assert "ground truth: healthy" in out
    assert "ground truth: hotspot" in out
    assert "ground truth: crack" in out
    assert "ground truth: dead_cell" in out
    # Every panel is previewed as ASCII and every decision names its source.
    assert "via rule" in out or "via model" in out
    assert "key features:" in out
    # The demo is honest about what the clear samples do and do not prove.
    assert "below 1.0" in out


def test_main_usage(capsys):
    assert cli.main() == 1
    assert "usage" in capsys.readouterr().out
