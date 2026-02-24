"""
Action DANN Domain Adaptation Data Collection

Grid: 4 batch x 2 optimisers x 4 learning rates x 4 precisions x 3 methods x 5 seeds = 960 runs

Usage:
    python action_dann_data_collector.py \
        --pykale_path /path/to/pykale \
        --dataset_root /path/to/video/data \
        --output_dir ./data \
        --job_id 0 \
        --total_jobs 32
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import psutil
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from yacs.config import CfgNode

warnings.filterwarnings('ignore', '.*does not have many workers.*')
warnings.filterwarnings('ignore', '.*Tensor Cores.*')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from causal_ai.data_collector import PyKaleCausalDataCollector
from examples.action_dann.utils.utils import ensure_dataset

from kale.loaddata.video_access import VideoDataset
from kale.loaddata.video_multi_domain import VideoBiDomainDatasets
from kale.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Domain pair configurations
DOMAIN_PAIR_MAP = {
    'EPIC_D1_D2': {
        'source': 'EPIC',
        'target': 'EPIC',
        'src_trainlist': 'epic_D1_train.pkl',
        'src_testlist': 'epic_D1_test.pkl',
        'tgt_trainlist': 'epic_D2_train.pkl',
        'tgt_testlist': 'epic_D2_test.pkl',
    },
    'EPIC_D1_D3': {
        'source': 'EPIC',
        'target': 'EPIC',
        'src_trainlist': 'epic_D1_train.pkl',
        'src_testlist': 'epic_D1_test.pkl',
        'tgt_trainlist': 'epic_D3_train.pkl',
        'tgt_testlist': 'epic_D3_test.pkl',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for the Action DANN data collection and experiment runner")
    parser.add_argument("--pykale_path", type=str, required=True, help="Path to the cloned PyKale repo")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory for video datasets (auto-downloads on first use)")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory where the csv files will be saved")
    parser.add_argument("--job_id", type=int, required=True, help="Index of the job (used for parallel runs on HPC")
    parser.add_argument("--total_jobs", type=int, required=True, help="Total number of jobs the experiment is split into")
    parser.add_argument("--devices", default="auto", help="Compute devices to use (e.g. 'cpu', 'gpu' or 'auto')")
    parser.add_argument("--domain_pair", type=str, default="EPIC_D1_D2",
                        choices=list(DOMAIN_PAIR_MAP.keys()),
                        help="Domain pair to use (default: EPIC_D1_D2)")
    parser.add_argument("--model_method", type=str, default="i3d",
                        choices=["r3d_18", "r2plus1d_18", "mc3_18", "i3d"],
                        help="Video backbone architecture (default: i3d)")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max epochs (default: use config value)")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run 1 batch of train/val/test to verify the pipeline works")
    return parser.parse_args()


def get_base_config(dataset_root, domain_pair_name, model_method):
    domain_pair = DOMAIN_PAIR_MAP[domain_pair_name]

    cfg = CfgNode()

    cfg.DATASET = CfgNode()
    cfg.DATASET.ROOT = dataset_root
    cfg.DATASET.SOURCE = domain_pair['source']
    cfg.DATASET.SRC_TRAINLIST = domain_pair['src_trainlist']
    cfg.DATASET.SRC_TESTLIST = domain_pair['src_testlist']
    cfg.DATASET.TARGET = domain_pair['target']
    cfg.DATASET.TGT_TRAINLIST = domain_pair['tgt_trainlist']
    cfg.DATASET.TGT_TESTLIST = domain_pair['tgt_testlist']
    cfg.DATASET.IMAGE_MODALITY = "rgb"
    cfg.DATASET.FRAMES_PER_SEGMENT = 8
    cfg.DATASET.NUM_REPEAT = 1
    cfg.DATASET.WEIGHT_TYPE = "natural"
    cfg.DATASET.SIZE_TYPE = "max"

    cfg.SOLVER = CfgNode()
    cfg.SOLVER.SEED = 2025
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.TYPE = "SGD"
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.NESTEROV = True
    cfg.SOLVER.MAX_EPOCHS = 10
    cfg.SOLVER.MIN_EPOCHS = 3
    cfg.SOLVER.TRAIN_BATCH_SIZE = 16
    cfg.SOLVER.LOG_EVERY_N_STEPS = 10
    cfg.SOLVER.AD_LAMBDA = True
    cfg.SOLVER.AD_LR = True
    cfg.SOLVER.INIT_LAMBDA = 1.0

    cfg.MODEL = CfgNode()
    cfg.MODEL.METHOD = model_method
    cfg.MODEL.ATTENTION = "None"

    cfg.DAN = CfgNode()
    cfg.DAN.METHOD = "CDAN"
    cfg.DAN.USERANDOM = False
    cfg.DAN.RANDOM_DIM = 1024

    cfg.OUTPUT = CfgNode()
    cfg.OUTPUT.VERBOSE = False
    cfg.OUTPUT.FAST_DEV_RUN = False
    cfg.OUTPUT.PB_FRESH = 0
    cfg.OUTPUT.OUT_DIR = "outputs"

    return cfg


def setup_dataset(cfg, seed):
    """Set up video dataset. Must be called per-seed as get_source_target uses seed internally."""
    try:
        source, target, num_classes = VideoDataset.get_source_target(
            VideoDataset(cfg.DATASET.SOURCE.upper()),
            VideoDataset(cfg.DATASET.TARGET.upper()),
            seed,
            cfg,
        )
        dataset = VideoBiDomainDatasets(
            source,
            target,
            image_modality=cfg.DATASET.IMAGE_MODALITY,
            seed=seed,
            config_weight_type=cfg.DATASET.WEIGHT_TYPE,
            config_size_type=cfg.DATASET.SIZE_TYPE,
        )

        logger.info(f"Dataset setup successful: {cfg.DATASET.SOURCE.upper()} -> {cfg.DATASET.TARGET.upper()}")
        return dataset, num_classes

    except Exception as e:
        logger.error(f"Dataset setup failed: {str(e)}")
        raise


def get_model(cfg, dataset, num_classes, pykale_path):
    try:
        sys.path.insert(0, os.path.join(pykale_path, "examples", "action_dann"))
        from model import get_model as get_action_model

        result = get_action_model(cfg, dataset, num_classes)

        if not isinstance(result, tuple) or len(result) != 2:
            logger.error(f"ERROR: get_action_model returned {type(result)}, expected (model, train_params)")
            raise ValueError(f"get_action_model returned unexpected type: {type(result)}")

        model, train_params = result
        logger.info(f"Model loaded: {type(model).__name__}")

        return model, train_params

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise


class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process()
        self.cpu_memory_baseline = self.process.memory_info().rss / 1024 / 1024
        self.cpu_memory_peak = self.cpu_memory_baseline
        self.gpu_available = torch.cuda.is_available()

        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def update(self):
        current_cpu = self.process.memory_info().rss / 1024 / 1024
        self.cpu_memory_peak = max(self.cpu_memory_peak, current_cpu)

    def get_cpu_peak(self):
        self.update()
        return self.cpu_memory_peak - self.cpu_memory_baseline

    def get_gpu_peak(self):
        if not self.gpu_available:
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024 / 1024


class EpochTimerCallback(pl.Callback):
    """Logs wall-clock time at the end of each training epoch."""

    def __init__(self):
        self.epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start
        epoch = trainer.current_epoch
        logger.info(f"Epoch {epoch + 1}/{trainer.max_epochs} completed in {elapsed:.1f}s")


def run_single_experiment(cfg, dataset, num_classes, pykale_path, collector,
                          fp_precision, optimiser_name, run_id, device, domain_pair_name,
                          fast_dev_run=False):
    try:
        if fp_precision == 'tf32':
            torch.set_float32_matmul_precision('medium')
        elif fp_precision == 'fp32-true':
            torch.set_float32_matmul_precision('highest')
        else:
            torch.set_float32_matmul_precision('high')

        set_seed(cfg.SOLVER.SEED)

        config_data = collector.capture_config(cfg, additional_params={
            'fp_precision': fp_precision,
            'run_id': run_id,
            'model_method': cfg.MODEL.METHOD,
            'image_modality': cfg.DATASET.IMAGE_MODALITY,
            'domain_pair': domain_pair_name,
            'num_classes': num_classes,
            'frames_per_segment': cfg.DATASET.FRAMES_PER_SEGMENT,
        })
        collector.log_config(config_data)

        memory_tracker = MemoryTracker()

        collector.start_timer('model_setup')
        model, train_params = get_model(cfg, dataset, num_classes, pykale_path)
        collector.end_timer('model_setup')
        memory_tracker.update()

        outdir = os.path.join(cfg.OUTPUT.OUT_DIR, f"run_{run_id}")
        os.makedirs(outdir, exist_ok=True)

        logger_pl = pl.loggers.TensorBoardLogger(outdir, name=f"run_{run_id}")

        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            monitor="valid_loss",
            mode="min",
        )

        if fp_precision == "tf32":
            precision_setting = "32-true"
        elif fp_precision == "bf16-mixed":
            precision_setting = "bf16-mixed"
        elif fp_precision == "fp16-mixed":
            precision_setting = "16-mixed"
        elif fp_precision == "fp32-true":
            precision_setting = "32-true"
        else:
            raise ValueError(f"Unknown precision: {fp_precision}")

        epoch_timer = EpochTimerCallback()

        trainer = pl.Trainer(
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            accelerator="auto" if device == "auto" else ("gpu" if device != "cpu" else "cpu"),
            devices=1 if device != "auto" else "auto",
            precision=precision_setting,
            callbacks=[checkpoint_callback, epoch_timer],
            logger=logger_pl,
            log_every_n_steps=cfg.SOLVER.LOG_EVERY_N_STEPS,
            enable_progress_bar=False,
            enable_model_summary=False,
            fast_dev_run=fast_dev_run,
        )

        collector.start_timer('training')
        trainer.fit(model)
        training_time = collector.end_timer('training')
        memory_tracker.update()

        training_metrics = collector.extract_trainer_metrics(trainer)
        collector.log_metrics(training_metrics)

        collector.start_timer('evaluation')
        test_results = trainer.test(ckpt_path="best")
        eval_time = collector.end_timer('evaluation')
        memory_tracker.update()

        test_metrics = collector.extract_trainer_metrics(trainer)
        collector.log_metrics(test_metrics)

        if test_results and isinstance(test_results, list) and len(test_results) > 0:
            test_dict = test_results[0]
            for key, value in test_dict.items():
                metric_key = f"test_{key}" if not key.startswith('test_') else key
                if isinstance(value, (int, float)):
                    collector.current_run[metric_key] = float(value)
                elif hasattr(value, 'item'):
                    collector.current_run[metric_key] = float(value.item())

        cpu_peak = memory_tracker.get_cpu_peak()
        gpu_peak = memory_tracker.get_gpu_peak()

        collector.log_memory_usage(cpu_peak, device='cpu')
        if torch.cuda.is_available():
            collector.log_memory_usage(gpu_peak, device='gpu')

        collector.save_run()

        logger.info(f"Run {run_id} completed | "
                   f"Train: {training_time:.1f}s | "
                   f"Eval: {eval_time:.1f}s | "
                   f"CPU: {cpu_peak:.1f}MB | "
                   f"GPU: {gpu_peak:.1f}MB | "
                   f"Precision: {fp_precision}")

        return True

    except Exception as e:
        logger.error(f"Run {run_id} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        collector.current_run['error'] = str(e)
        collector.save_run()
        return False


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info(f"Action DANN Data Collection - Job {args.job_id}/{args.total_jobs}")
    logger.info("=" * 80)

    optimiser_lr_map = {
        'AdamW': [3e-4, 1e-3],
        'SGD': [0.03, 0.1],
    }

    param_grid = {
        'batch_size': [4, 8, 16, 32],
        'optimiser': ['AdamW', 'SGD'],
        'fp_precision': ['tf32', 'bf16-mixed', 'fp16-mixed', 'fp32-true'],
        'adaptation_method': ['DAN', 'DANN', 'CDAN'],
        'seed': [1, 7, 42, 123, 2023],
    }

    all_combinations = []
    for batch_size in param_grid['batch_size']:
        for optimiser in param_grid['optimiser']:
            for learning_rate in optimiser_lr_map[optimiser]:
                for fp_precision in param_grid['fp_precision']:
                    for adaptation_method in param_grid['adaptation_method']:
                        for seed in param_grid['seed']:
                            all_combinations.append({
                                'batch_size': batch_size,
                                'optimiser': optimiser,
                                'learning_rate': learning_rate,
                                'fp_precision': fp_precision,
                                'adaptation_method': adaptation_method,
                                'seed': seed,
                            })

    total_runs = len(all_combinations)

    runs_per_job = (total_runs + args.total_jobs - 1) // args.total_jobs
    start_idx = args.job_id * runs_per_job
    end_idx = min(start_idx + runs_per_job, total_runs)

    logger.info(f"Total experiments: {total_runs}")
    logger.info(f"Domain pair: {args.domain_pair}")
    logger.info(f"Model method: {args.model_method}")
    logger.info(f"This job handles: runs {start_idx} to {end_idx - 1} ({end_idx - start_idx} runs)")

    output_csv = Path(args.output_dir) / f"action_results_job{args.job_id:03d}.csv"
    collector = PyKaleCausalDataCollector(str(output_csv))

    ensure_dataset(args.dataset_root)

    pykale_path = Path(args.pykale_path)

    start_time = time.time()
    successful_runs = 0
    failed_runs = 0

    for idx in range(start_idx, end_idx):
        params = all_combinations[idx]

        cfg = get_base_config(args.dataset_root, args.domain_pair, args.model_method)
        cfg.defrost()

        cfg.SOLVER.BASE_LR = params['learning_rate']
        cfg.SOLVER.TRAIN_BATCH_SIZE = params['batch_size']
        cfg.DAN.METHOD = params['adaptation_method']
        cfg.SOLVER.SEED = params['seed']
        if args.max_epochs is not None:
            cfg.SOLVER.MAX_EPOCHS = args.max_epochs

        cfg.SOLVER.TYPE = params['optimiser']
        cfg.SOLVER.WEIGHT_DECAY = 0.0005

        if params['optimiser'] == 'SGD':
            cfg.SOLVER.MOMENTUM = 0.9
            cfg.SOLVER.NESTEROV = True
        else:
            cfg.SOLVER.MOMENTUM = 0.0
            cfg.SOLVER.NESTEROV = False

        cfg.freeze()

        # Dataset must be recreated per seed (VideoDataset.get_source_target uses seed internally)
        try:
            dataset, num_classes = setup_dataset(cfg, params['seed'])
        except Exception as e:
            logger.error(f"Failed to setup dataset for run {idx}: {str(e)}")
            failed_runs += 1
            continue

        logger.info("=" * 80)
        logger.info(f"Run {idx} ({idx + 1}/{total_runs})")
        logger.info(f"Method: {params['adaptation_method']}, Precision: {params['fp_precision']}, Seed: {params['seed']}")
        logger.info(f"Batch: {params['batch_size']}, Optimizer: {params['optimiser']}, LR: {params['learning_rate']}")
        logger.info(f"Domain: {args.domain_pair}, Model: {args.model_method}")
        logger.info("=" * 80)

        success = run_single_experiment(
            cfg=cfg,
            dataset=dataset,
            num_classes=num_classes,
            pykale_path=str(pykale_path),
            collector=collector,
            fp_precision=params['fp_precision'],
            optimiser_name=params['optimiser'],
            run_id=idx,
            device=args.devices,
            domain_pair_name=args.domain_pair,
            fast_dev_run=args.fast_dev_run,
        )

        if success:
            successful_runs += 1
        else:
            failed_runs += 1

        # Clear GPU cache between runs (video models consume much more memory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"Job {args.job_id} completed")
    logger.info(f"Runs: {end_idx - start_idx} | Success: {successful_runs} | Failed: {failed_runs}")
    logger.info(f"Time: {elapsed / 60:.1f} minutes")

    output_path = collector.export_data()
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
