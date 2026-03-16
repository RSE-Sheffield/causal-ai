"""Tests for causal_ai.__main__ CLI."""

import json
import subprocess
import sys

import pytest


class TestSummaryCommand:
    def test_summary_output(self, cluster_dir):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "summary", str(cluster_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Total tests:" in result.stdout
        assert "Passed:" in result.stdout

    def test_summary_json_flag(self, cluster_dir):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "summary", str(cluster_dir), "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # The JSON block should be valid
        json_start = result.stdout.index("{")
        parsed = json.loads(result.stdout[json_start:])
        assert "total" in parsed
        assert "passed" in parsed

    def test_summary_invalid_dir(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "summary", str(tmp_path / "nope")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_summary_no_results(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "summary", str(empty)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestCompareCommand:
    def test_compare_output(self, multi_cluster_dir):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "compare", str(multi_cluster_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "[stanage]" in result.stdout
        assert "[bede]" in result.stdout

    def test_compare_shows_divergent(self, multi_cluster_dir):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "compare", str(multi_cluster_dir)],
            capture_output=True, text=True,
        )
        assert "Divergent tests" in result.stdout or "divergent" in result.stdout.lower()

    def test_compare_json_flag(self, multi_cluster_dir):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "compare", str(multi_cluster_dir), "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        json_start = result.stdout.index("{")
        parsed = json.loads(result.stdout[json_start:])
        assert "per_cluster" in parsed
        assert "divergent_tests" in parsed

    def test_compare_no_subdirs(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "compare", str(empty)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestHelpAndArgs:
    def test_no_command_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "causal_ai", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "summary" in result.stdout
        assert "compare" in result.stdout
