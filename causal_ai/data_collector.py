"""
A custom data collector wrapper to capture experimental runs from PyKale: https://pykale.github.io/
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PyKaleCausalDataCollector:
    """Collects runtime data from PyKale pipeline API for causal testing"""

    def __init__(self, output_path: str = "runtime_data.csv"):
        """Initialise the causal data collector

        Args:
            output_path: Path where the CSV file will be saved
        """
        self.output_path = Path(output_path)
        self.data_records: List[Dict[str, Any]] = []
        self.current_run: Dict[str, Any] = {}
        self.stage_timers: Dict[str, float] = {}

    def start_timer(self, stage_name: str) -> None:
        """Start timing a workflow stage."""
        self.stage_timers[f"{stage_name}_start"] = time.time()

    def end_timer(self, stage_name: str) -> float:
        """End timing a workflow stage and return elapsed time in seconds."""
        start_key = f"{stage_name}_start"
        if start_key not in self.stage_timers:
            logger.warning(f"Timer for stage '{stage_name}' was not started")
            return 0.0

        elapsed = time.time() - self.stage_timers[start_key]
        self.current_run[f"{stage_name}_time_seconds"] = elapsed
        return elapsed

    def capture_config(self, cfg, additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract the causal input variables from config object script

        Args:
            cfg: YACS CfgNode configuration object from PyKale
            additional_params: Additional parameters (e.g fp_precision, run_id)

        Returns:
            Dictionary of configuration parameters
        """
        config_data = {
            "learning_rate": float(cfg.SOLVER.BASE_LR),
            "batch_size": int(cfg.SOLVER.TRAIN_BATCH_SIZE),
            "optimiser_type": str(cfg.SOLVER.TYPE) if hasattr(cfg.SOLVER, 'TYPE') else None,
            "adaptation_method": str(cfg.DAN.METHOD) if hasattr(cfg, 'DAN') else None,
            "seed": int(cfg.SOLVER.SEED) if hasattr(cfg.SOLVER, 'SEED') else None,
        }
        # Update the config data if additional params exist
        if additional_params:
            config_data.update(additional_params)

        return config_data

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters for the current run."""
        self.current_run.update(config)

    def extract_trainer_metrics(self, trainer) -> Dict[str, Any]:
        """Extract metrics from PyTorch Lightning trainer after training/testing

        Args:
            trainer: PyTorch Lightning Trainer instance

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}

        callback_metrics = trainer.callback_metrics if hasattr(trainer, 'callback_metrics') else {}
        logged_metrics = trainer.logged_metrics if hasattr(trainer, 'logged_metrics') else {}
        all_metrics = {**callback_metrics, **logged_metrics}

        metric_mapping = {
            "train_total_loss": "train_total_loss",
            "train_task_loss": "train_task_loss",
            "train_domain_div_loss": "train_domain_div_loss",
            "valid_loss": "valid_loss",
            "valid_task_loss": "valid_task_loss",
            "valid_domain_div_loss": "valid_domain_div_loss",
            "test_loss": "test_loss",
            "test_task_loss": "test_task_loss",
            "test_domain_div_loss": "test_domain_div_loss",
        }

        for key, target_key in metric_mapping.items():
            if key in all_metrics:
                value = all_metrics[key]
                if hasattr(value, 'item'):
                    metrics[target_key] = float(value.item())
                elif isinstance(value, (int, float)):
                    metrics[target_key] = float(value)
                elif isinstance(value, list) and len(value) > 0:
                    last_val = value[-1]
                    metrics[target_key] = float(last_val.item()) if hasattr(last_val, 'item') else float(last_val)

        return metrics

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for the current run"""
        self.current_run.update(metrics)

    def log_memory_usage(self, peak_memory_mb: float, device: str = "cpu") -> None:
        """Log peak memory usage

        Args:
            peak_memory_mb: Peak memory usage in megabytes
            device: Device type ('cpu' or 'gpu')
        """
        if device == "cpu":
            self.current_run["memory_peak_mb"] = peak_memory_mb
        elif device == "gpu":
            self.current_run["gpu_memory_peak_mb"] = peak_memory_mb

    def save_run(self) -> None:
        """Save the current run and reset for next run"""
        if self.current_run:
            self.data_records.append(self.current_run.copy())
            logger.info(f"Saved run {len(self.data_records)}: seed={self.current_run.get('seed', 'N/A')}")
            self.current_run = {}
            self.stage_timers = {}
        else:
            logger.warning("No data to save for current run")

    def export_data(self) -> Path:
        """Export all collected data to CSV

        Returns:
            Path to the exported CSV file
        """
        if not self.data_records:
            logger.warning("No data records to export")
            return self.output_path

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.data_records)
        df.to_csv(self.output_path, index=False)

        logger.info(f"Exported {len(df)} runs to {self.output_path}")
        logger.info(f"Columns captured: {list(df.columns)}")

        logger.info("\n=== Data summary ===")
        logger.info(f"Total runs: {len(df)}")
        logger.info(f"Unique variables combinations")
        for col in df.columns:
            if col.endswith('_time_seconds') or col.endswith('_mb') or 'loss' in col or 'accuracy' in col:
                continue
            unique_vals = df[col].nunique()
            if unique_vals < 20:
                logger.info(f"  {col}: there are {unique_vals} unique values")

        return self.output_path

    def get_dataframe(self) -> pd.DataFrame:
        """Get collected data as a pandas DataFrame."""
        return pd.DataFrame(self.data_records)

    def checkpoint_save(self, checkpoint_path: Optional[str] = None) -> None:
        """Save a checkpoint of current data collection progress

        Args:
            checkpoint_path: Optional path for checkpoint file
        """
        if checkpoint_path is None:
            checkpoint_path = str(self.output_path).replace('.csv', '.checkpoint.csv')

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.data_records)
        df.to_csv(checkpoint_path, index=False)
        logger.info(f"Checkpoint saved: {len(df)} runs to {checkpoint_path}")
