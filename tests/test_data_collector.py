"""Tests for causal_ai.data_collector.PyKaleCausalDataCollector."""

import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from causal_ai.data_collector import PyKaleCausalDataCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides):
    """Build a minimal YACS-like config namespace for testing."""
    solver = SimpleNamespace(
        BASE_LR=0.001,
        TRAIN_BATCH_SIZE=64,
        TYPE="AdamW",
        SEED=42,
    )
    dan = SimpleNamespace(METHOD="DANN")
    cfg = SimpleNamespace(SOLVER=solver, DAN=dan)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_trainer(callback_metrics=None, logged_metrics=None):
    """Build a minimal trainer-like object."""
    return SimpleNamespace(
        callback_metrics=callback_metrics or {},
        logged_metrics=logged_metrics or {},
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_output_path(self):
        c = PyKaleCausalDataCollector()
        assert c.output_path == Path("runtime_data.csv")

    def test_custom_output_path(self, tmp_path):
        p = str(tmp_path / "out.csv")
        c = PyKaleCausalDataCollector(p)
        assert c.output_path == Path(p)

    def test_empty_state(self):
        c = PyKaleCausalDataCollector()
        assert c.data_records == []
        assert c.current_run == {}
        assert c.stage_timers == {}


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------

class TestTimers:
    def test_start_and_end_timer(self):
        c = PyKaleCausalDataCollector()
        c.start_timer("training")
        time.sleep(0.05)
        elapsed = c.end_timer("training")
        assert elapsed >= 0.04
        assert "training_time_seconds" in c.current_run

    def test_end_timer_without_start_returns_zero(self):
        c = PyKaleCausalDataCollector()
        assert c.end_timer("missing") == 0.0


# ---------------------------------------------------------------------------
# Config capture
# ---------------------------------------------------------------------------

class TestCaptureConfig:
    def test_basic_capture(self):
        cfg = _make_cfg()
        c = PyKaleCausalDataCollector()
        result = c.capture_config(cfg)

        assert result["learning_rate"] == 0.001
        assert result["batch_size"] == 64
        assert result["optimiser_type"] == "AdamW"
        assert result["adaptation_method"] == "DANN"
        assert result["seed"] == 42

    def test_additional_params_merged(self):
        cfg = _make_cfg()
        c = PyKaleCausalDataCollector()
        result = c.capture_config(cfg, additional_params={"fp_precision": "fp16"})
        assert result["fp_precision"] == "fp16"

    def test_missing_optional_attrs(self):
        solver = SimpleNamespace(BASE_LR=0.01, TRAIN_BATCH_SIZE=32)
        cfg = SimpleNamespace(SOLVER=solver)
        c = PyKaleCausalDataCollector()
        result = c.capture_config(cfg)

        assert result["optimiser_type"] is None
        assert result["adaptation_method"] is None
        assert result["seed"] is None

    def test_log_config_updates_current_run(self):
        c = PyKaleCausalDataCollector()
        c.log_config({"learning_rate": 0.01, "batch_size": 128})
        assert c.current_run["learning_rate"] == 0.01
        assert c.current_run["batch_size"] == 128


# ---------------------------------------------------------------------------
# Trainer metric extraction
# ---------------------------------------------------------------------------

class TestExtractTrainerMetrics:
    def test_extracts_float_metrics(self):
        trainer = _make_trainer(callback_metrics={
            "train_total_loss": 0.5,
            "test_loss": 0.3,
        })
        c = PyKaleCausalDataCollector()
        metrics = c.extract_trainer_metrics(trainer)
        assert metrics["train_total_loss"] == 0.5
        assert metrics["test_loss"] == 0.3

    def test_extracts_tensor_like_metrics(self):
        """Simulate a tensor with an .item() method."""
        class FakeTensor:
            def item(self):
                return 1.23
        trainer = _make_trainer(callback_metrics={"valid_loss": FakeTensor()})
        c = PyKaleCausalDataCollector()
        metrics = c.extract_trainer_metrics(trainer)
        assert metrics["valid_loss"] == pytest.approx(1.23)

    def test_extracts_list_metrics(self):
        trainer = _make_trainer(callback_metrics={"test_task_loss": [0.9, 0.5, 0.2]})
        c = PyKaleCausalDataCollector()
        metrics = c.extract_trainer_metrics(trainer)
        assert metrics["test_task_loss"] == 0.2  # last value

    def test_ignores_unknown_keys(self):
        trainer = _make_trainer(callback_metrics={"unknown_metric": 99.0})
        c = PyKaleCausalDataCollector()
        metrics = c.extract_trainer_metrics(trainer)
        assert "unknown_metric" not in metrics

    def test_handles_missing_attrs(self):
        trainer = SimpleNamespace()  # no callback_metrics or logged_metrics
        c = PyKaleCausalDataCollector()
        metrics = c.extract_trainer_metrics(trainer)
        assert metrics == {}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_metrics(self):
        c = PyKaleCausalDataCollector()
        c.log_metrics({"test_loss": 0.1, "test_target_acc": 0.95})
        assert c.current_run["test_loss"] == 0.1

    def test_log_memory_cpu(self):
        c = PyKaleCausalDataCollector()
        c.log_memory_usage(512.0, device="cpu")
        assert c.current_run["memory_peak_mb"] == 512.0

    def test_log_memory_gpu(self):
        c = PyKaleCausalDataCollector()
        c.log_memory_usage(1024.0, device="gpu")
        assert c.current_run["gpu_memory_peak_mb"] == 1024.0

    def test_log_dataset_info(self):
        c = PyKaleCausalDataCollector()
        c.log_dataset_info(train_size=1000, valid_size=200, test_size=300)
        assert c.current_run["train_dataset_size"] == 1000
        assert c.current_run["valid_dataset_size"] == 200
        assert c.current_run["test_dataset_size"] == 300


# ---------------------------------------------------------------------------
# Save / export / checkpoint
# ---------------------------------------------------------------------------

class TestSaveAndExport:
    def test_save_run_appends_and_resets(self):
        c = PyKaleCausalDataCollector()
        c.log_config({"seed": 1})
        c.save_run()

        assert len(c.data_records) == 1
        assert c.current_run == {}
        assert c.stage_timers == {}

    def test_save_empty_run_does_nothing(self):
        c = PyKaleCausalDataCollector()
        c.save_run()
        assert len(c.data_records) == 0

    def test_export_creates_csv(self, tmp_path):
        out = tmp_path / "data.csv"
        c = PyKaleCausalDataCollector(str(out))
        c.log_config({"seed": 1, "batch_size": 64})
        c.log_metrics({"test_loss": 0.2})
        c.save_run()

        path = c.export_data()

        assert path == out
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 1
        assert "seed" in df.columns
        assert "test_loss" in df.columns

    def test_export_empty_returns_path(self, tmp_path):
        out = tmp_path / "empty.csv"
        c = PyKaleCausalDataCollector(str(out))
        path = c.export_data()
        assert path == out
        assert not out.exists()

    def test_export_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "data.csv"
        c = PyKaleCausalDataCollector(str(out))
        c.log_config({"seed": 1})
        c.save_run()
        c.export_data()
        assert out.exists()

    def test_multiple_runs(self, tmp_path):
        out = tmp_path / "multi.csv"
        c = PyKaleCausalDataCollector(str(out))
        for seed in [1, 7, 42]:
            c.log_config({"seed": seed})
            c.log_metrics({"test_loss": seed * 0.1})
            c.save_run()

        c.export_data()
        df = pd.read_csv(out)
        assert len(df) == 3
        assert list(df["seed"]) == [1, 7, 42]

    def test_get_dataframe(self):
        c = PyKaleCausalDataCollector()
        c.log_config({"seed": 1})
        c.save_run()
        df = c.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_checkpoint_save(self, tmp_path):
        out = tmp_path / "data.csv"
        c = PyKaleCausalDataCollector(str(out))
        c.log_config({"seed": 1})
        c.save_run()
        c.checkpoint_save()

        checkpoint = tmp_path / "data.checkpoint.csv"
        assert checkpoint.exists()
        df = pd.read_csv(checkpoint)
        assert len(df) == 1

    def test_checkpoint_custom_path(self, tmp_path):
        c = PyKaleCausalDataCollector(str(tmp_path / "data.csv"))
        c.log_config({"seed": 1})
        c.save_run()

        custom = tmp_path / "custom_checkpoint.csv"
        c.checkpoint_save(str(custom))
        assert custom.exists()
