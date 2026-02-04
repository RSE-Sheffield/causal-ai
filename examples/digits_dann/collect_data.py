"""
DANN Domain Adaptation Data Collection

Grid: 4 batch x 2 optimisers x 4 learning rates x 4 precisions x 3 methods x 5 seeds = 960 runs

Usage:
    python dann_data_collector_fixed_expanded.py \
        --pykale_path /path/to/pykale \
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

from kale.loaddata.image_access import DigitDataset
from kale.loaddata.multi_domain import MultiDomainAccess, MultiDomainDataset
from kale.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for the DANN data collection and experiment runner")
    parser.add_argument("--pykale_path", type=str, required=True, help="Path to the cloned PyKale repo")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory where the csv files will be saved")
    parser.add_argument("--job_id", type=int, required=True, help="Index of the job (used for parallel runs on HPC")
    parser.add_argument("--total_jobs", type=int, required=True, help="Total number of jobs the experiment is split into")
    parser.add_argument("--devices", default="auto", help="Compute devices to use (e.g. 'cpu', 'gpu' or 'auto')")
    return parser.parse_args()


def get_base_config():
    cfg = CfgNode()

    cfg.DATASET = CfgNode()
    cfg.DATASET.ROOT = "./data"
    cfg.DATASET.NAME = "digits"
    cfg.DATASET.SOURCE = "mnist"
    cfg.DATASET.TARGET = "usps"
    cfg.DATASET.NUM_CLASSES = 10
    cfg.DATASET.NUM_REPEAT = 1
    cfg.DATASET.VALID_SPLIT_RATIO = 0.1
    cfg.DATASET.DIMENSION = 784
    cfg.DATASET.WEIGHT_TYPE = "natural"
    cfg.DATASET.SIZE_TYPE = "source"

    cfg.SOLVER = CfgNode()
    cfg.SOLVER.SEED = 2025
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.TYPE = "SGD"
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.NESTEROV = True
    cfg.SOLVER.MAX_EPOCHS = 20
    cfg.SOLVER.MIN_EPOCHS = 5
    cfg.SOLVER.NUM_WORKERS = 0
    cfg.SOLVER.TRAIN_BATCH_SIZE = 128
    cfg.SOLVER.TEST_BATCH_SIZE = 200
    cfg.SOLVER.LOG_EVERY_N_STEPS = 10
    cfg.SOLVER.AD_LAMBDA = True
    cfg.SOLVER.AD_LR = True
    cfg.SOLVER.INIT_LAMBDA = 1.0

    cfg.DAN = CfgNode()
    cfg.DAN.METHOD = "CDAN"
    cfg.DAN.USERANDOM = False
    cfg.DAN.RANDOM_DIM = 1024

    cfg.OUTPUT = CfgNode()
    cfg.OUTPUT.VERBOSE = False
    cfg.OUTPUT.PB_FRESH = 0
    cfg.OUTPUT.OUT_DIR = "outputs"

    cfg.COMET = CfgNode()
    cfg.COMET.ENABLE = False

    cfg.OPTIMIZER = CfgNode()
    cfg.OPTIMIZER.TYPE = "SGD"
    cfg.OPTIMIZER.OPTIM_PARAMS = CfgNode()
    cfg.OPTIMIZER.OPTIM_PARAMS.MOMENTUM = 0.9
    cfg.OPTIMIZER.OPTIM_PARAMS.WEIGHT_DECAY = 0.0005
    cfg.OPTIMIZER.OPTIM_PARAMS.NESTEROV = True

    return cfg


def setup_dataset(cfg, pykale_path):
    data_root = os.path.join(pykale_path, "examples", "digits_dann", "data")
    cfg.defrost()
    cfg.DATASET.ROOT = data_root
    cfg.freeze()

    try:
        data_src = DigitDataset(cfg.DATASET.SOURCE.upper())
        data_tgt = DigitDataset(cfg.DATASET.TARGET.upper())
        num_channels = max(
            DigitDataset.get_channel_numbers(data_src),
            DigitDataset.get_channel_numbers(data_tgt)
        )

        data_access = MultiDomainAccess(
            {
                cfg.DATASET.SOURCE.upper(): DigitDataset.get_access(data_src, cfg.DATASET.ROOT)[0],
                cfg.DATASET.TARGET.upper(): DigitDataset.get_access(data_tgt, cfg.DATASET.ROOT)[0],
            },
            cfg.DATASET.NUM_CLASSES,
            return_domain_label=True,
        )
        dataset = MultiDomainDataset(data_access)
        
        logger.info(f"Dataset setup successful: {cfg.DATASET.SOURCE.upper()} -> {cfg.DATASET.TARGET.upper()}")
        return dataset, num_channels
    
    except Exception as e:
        logger.error(f"Dataset setup failed: {str(e)}")
        raise


def get_model(cfg, dataset, num_channels, pykale_path):
    try:
        sys.path.insert(0, os.path.join(pykale_path, "examples", "digits_dann"))
        from model import get_model as get_dann_model
        
        result = get_dann_model(cfg, dataset, num_channels)
        
        if not isinstance(result, tuple) or len(result) != 2:
            logger.error(f"ERROR: get_dann_model returned {type(result)}, expected (model, train_params)")
            raise ValueError(f"get_dann_model returned unexpected type: {type(result)}")
        
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


def run_single_experiment(cfg, dataset, num_channels, pykale_path, collector, fp_precision, optimiser_name, run_id, device):
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
            'run_id': run_id
        })
        collector.log_config(config_data)

        memory_tracker = MemoryTracker()

        collector.start_timer('model_setup')
        model, train_params = get_model(cfg, dataset, num_channels, pykale_path)
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

        trainer = pl.Trainer(
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            accelerator="auto" if device == "auto" else ("gpu" if device != "cpu" else "cpu"),
            devices=1 if device != "auto" else "auto",
            precision=precision_setting,
            callbacks=[checkpoint_callback],
            logger=logger_pl,
            log_every_n_steps=cfg.SOLVER.LOG_EVERY_N_STEPS,
            enable_progress_bar=False,
            enable_model_summary=False,
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
    logger.info(f"DANN Data Collection - Job {args.job_id}/{args.total_jobs}")
    logger.info("=" * 80)

    optimiser_lr_map = {
        'AdamW': [3e-4, 1e-3],
        'SGD': [0.03, 0.1],
    }

    param_grid = {
        'batch_size': [64, 128, 256, 512],
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
    logger.info(f"This job handles: runs {start_idx} to {end_idx-1} ({end_idx-start_idx} runs)")

    output_csv = Path(args.output_dir) / f"production_results_job{args.job_id:03d}.csv"
    collector = PyKaleCausalDataCollector(str(output_csv))

    pykale_path = Path(args.pykale_path)

    start_time = time.time()
    successful_runs = 0
    failed_runs = 0

    for idx in range(start_idx, end_idx):
        params = all_combinations[idx]

        cfg = get_base_config()
        cfg.defrost()

        cfg.SOLVER.BASE_LR = params['learning_rate']
        cfg.SOLVER.TRAIN_BATCH_SIZE = params['batch_size']
        cfg.DAN.METHOD = params['adaptation_method']
        cfg.SOLVER.SEED = params['seed']

        cfg.SOLVER.TYPE = params['optimiser']
        cfg.SOLVER.WEIGHT_DECAY = 0.0005
        cfg.OPTIMIZER.TYPE = params['optimiser']
        cfg.OPTIMIZER.OPTIM_PARAMS.WEIGHT_DECAY = 0.0005

        if params['optimiser'] == 'SGD':
            cfg.SOLVER.MOMENTUM = 0.9
            cfg.SOLVER.NESTEROV = True
            cfg.OPTIMIZER.OPTIM_PARAMS.MOMENTUM = 0.9
            cfg.OPTIMIZER.OPTIM_PARAMS.NESTEROV = True
        else:
            cfg.SOLVER.MOMENTUM = 0.0
            cfg.SOLVER.NESTEROV = False
            cfg.OPTIMIZER.OPTIM_PARAMS.MOMENTUM = 0.0
            cfg.OPTIMIZER.OPTIM_PARAMS.NESTEROV = False

        cfg.freeze()

        try:
            dataset, num_channels = setup_dataset(cfg, str(pykale_path))
        except Exception as e:
            logger.error(f"Failed to setup dataset for run {idx}: {str(e)}")
            failed_runs += 1
            continue

        logger.info(f"[{idx+1}/{total_runs}] Run {idx}: "
                   f"LR={params['learning_rate']:.0e}, BS={params['batch_size']}, "
                   f"Opt={params['optimiser']}, FP={params['fp_precision']}, "
                   f"Method={params['adaptation_method']}, Seed={params['seed']}")

        success = run_single_experiment(
            cfg=cfg,
            dataset=dataset,
            num_channels=num_channels,
            pykale_path=str(pykale_path),
            collector=collector,
            fp_precision=params['fp_precision'],
            optimiser_name=params['optimiser'],
            run_id=idx,
            device=args.devices,
        )

        # Tracka all runs
        if success:
            successful_runs += 1
        else:
            failed_runs += 1

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"Job {args.job_id} completed")
    logger.info(f"Runs: {end_idx - start_idx} | Success: {successful_runs} | Failed: {failed_runs}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")

    output_path = collector.export_data()
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
