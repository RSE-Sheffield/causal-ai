"""Tests for causal_ai.merge_csv."""

import pytest
import pandas as pd

from causal_ai.utils import merge_csv_files


def _write_csv(path, rows, columns=("run_id", "seed", "test_loss")):
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)


class TestMergeCsvFiles:
    def test_merges_multiple_files(self, tmp_path):
        _write_csv(tmp_path / "job0.csv", [(0, 1, 0.5), (1, 7, 0.4)])
        _write_csv(tmp_path / "job1.csv", [(2, 42, 0.3)])

        merged = merge_csv_files(tmp_path)
        assert len(merged) == 3
        assert list(merged["run_id"]) == [0, 1, 2]

    def test_sorts_by_run_id(self, tmp_path):
        _write_csv(tmp_path / "job1.csv", [(5, 1, 0.1)])
        _write_csv(tmp_path / "job0.csv", [(2, 7, 0.2)])

        merged = merge_csv_files(tmp_path)
        assert list(merged["run_id"]) == [2, 5]

    def test_works_without_run_id(self, tmp_path):
        df = pd.DataFrame({"seed": [1, 2], "loss": [0.5, 0.3]})
        df.to_csv(tmp_path / "job0.csv", index=False)

        merged = merge_csv_files(tmp_path)
        assert len(merged) == 2

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_csv_files(tmp_path)
