# Action DANN Domain Adaptation - Production Data Collection

Production-ready causal testing grid for video domain adaptation (EPIC D1 to D2).

## Dataset setup

The EPIC-Kitchens data lives under the path you pass as `--dataset_root`
(e.g. `/mnt/parscratch/users/cs1fxa/datasets/EgoAction`). Two things need
to be there: annotation pkl files and extracted video frames.

### 1. Annotations (handled automatically)

The first time `action_dann_data_collector.py` runs it calls
`ensure_dataset(dataset_root)`, which downloads:

- **D1** annotations from the
  [pykale/data](https://github.com/pykale/data) test-data zip
- **D2 and D3** annotations from the
  [MM-SADA Domain Adaptation Splits](https://github.com/jonmun/MM-SADA_Domain_Adaptation_Splits)

The pkl files are placed at
`<dataset_root>/EPIC/EPIC_KITCHENS_2018/annotations/labels_train_test/`.
No manual action needed — just make sure the machine has internet access on
the first run.

### 2. Video frames (manual download)

PyKale loads individual JPEG frames from disk. These are several GB so they
aren't downloaded automatically. Run the provided script once on HPC:

```bash
bash utils/download_frames.sh /mnt/parscratch/users/cs1fxa/datasets/EgoAction
```

This fetches pre-extracted RGB frames for P08 (D1), P01 (D2) and P22 (D3)
from the Bristol Research Data Portal, extracts the tar archives, and places
them in the layout PyKale expects. The script is resumable — videos that
have already been extracted are skipped.

Only RGB frames are downloaded by default (matching `image_modality=rgb` in
the config). Edit `MODALITIES` at the top of the script if you also need
optical flow.

```
<dataset_root>/EPIC/EPIC_KITCHENS_2018/
    frames_rgb_flow/
        rgb/
            train/
                P01/P01_01/frame_0000000001.jpg ...
                P08/P08_01/frame_0000000001.jpg ...
                P22/P22_01/frame_0000000001.jpg ...
            test/
                ...
        flow/
            train/ ...
            test/ ...
    annotations/
        labels_train_test/
            epic_D1_train.pkl   epic_D1_test.pkl
            epic_D2_train.pkl   epic_D2_test.pkl
            epic_D3_train.pkl   epic_D3_test.pkl
```

This only needs to be done once. Subsequent runs will reuse the data.

### 3. Verify the setup

```bash
# check annotations exist
ls /mnt/parscratch/users/cs1fxa/datasets/EgoAction/EPIC/EPIC_KITCHENS_2018/annotations/labels_train_test/

# check a few frames exist
ls /mnt/parscratch/users/cs1fxa/datasets/EgoAction/EPIC/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/P08/P08_01/ | head
```

## Configuration

### Precision Types
- **tf32**: TensorFloat-32 (H100/A100 default, automatic optimisation)
- **bf16-mixed**: BFloat16 mixed precision (modern standard, explicit mixed precision)
- **fp16-mixed**: FP16+FP32 mixed precision (de facto standard)
- **fp32-true**: True full precision baseline (causal control, highest precision)

### Optimisers with Learning Rates
- **AdamW**: 3e-4, 1e-3 (modern baseline with decoupled weight decay)
- **SGD**: 0.03, 0.1 (classical baseline)

### Other Parameters
- **Batch sizes**: 4, 8, 16, 32 (video workloads need smaller batches)
- **Methods**: DAN, DANN, CDAN (validated domain adaptation methods)
- **Seeds**: 1, 7, 42, 123, 2023 (5 seeds for variance estimation)

### Total Experiments
**960 runs** = 4 batch sizes x 2 optimisers x 2 LRs each x 4 precisions x 3 methods x 5 seeds

### Fixed Parameters
- **Model method**: i3d (configurable via `--model_method` CLI arg)
- **Domain pair**: EPIC D1 to D2 (configurable via `--domain_pair` CLI arg)
- **Image modality**: rgb
- **Frames per segment**: 16

## Usage

### Submit Jobs

```bash
sbatch submit_action_jobs.sh
```

This runs 32 parallel jobs on H100 GPUs, each handling 30 experiments.

### Monitor Progress

```bash
# Check job status
squeue -u cs1fxa

# Watch output
tail -f logs/action_job_*_0.out

# Count completed jobs
ls data/production/*.csv | wc -l  # Should reach 32
```

### Merge Results

After all jobs complete:

```bash
python merge_results.py \
    --input_dir ./data/production \
    --output ./data/action_results_complete.csv
```

### Verify Data

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/action_results_complete.csv')

print(f'Total runs: {len(df)}')
print(f'\nPrecision distribution:')
print(df['fp_precision'].value_counts())
print(f'\nOptimiser distribution:')
print(df['optimiser_type'].value_counts())
print(f'\nBatch size distribution:')
print(df['batch_size'].value_counts())
print(f'\nMethod distribution:')
print(df['adaptation_method'].value_counts())
print(f'\nSeed distribution:')
print(df['seed'].value_counts())
print(f'\nLearning rates:')
print(df['learning_rate'].value_counts())

if 'error' in df.columns:
    failed = df['error'].notna().sum()
    print(f'\nFailed runs: {failed}')
    if failed > 0:
        print(df[df['error'].notna()]['fp_precision'].value_counts())
"
```

Expected output:
```
Total runs: 960

Precision distribution:
tf32         240
fp32-true    240
fp16-mixed   240
bf16-mixed   240

Optimiser distribution:
AdamW    480
SGD      480

Batch size distribution:
32     240
16     240
8      240
4      240

Method distribution:
DAN     320
DANN    320
CDAN    320

Seed distribution:
2023    192
123     192
42      192
7       192
1       192

Learning rates:
0.1000    240
0.0300    240
0.0010    240
0.0003    240
```

## Expected CSV Schema

Each row in the output CSV contains:

### Input Variables (Causes)
| Column | Type | Description |
|--------|------|-------------|
| `learning_rate` | float | Learning rate |
| `batch_size` | int | Training batch size (4, 8, 16, 32) |
| `optimiser_type` | str | Optimiser type (AdamW, SGD) |
| `adaptation_method` | str | DA method (DAN, DANN, CDAN) |
| `seed` | int | Random seed |
| `fp_precision` | str | Precision type (tf32, bf16-mixed, fp16-mixed, fp32-true) |
| `run_id` | int | Global run index |
| `model_method` | str | Video backbone (fixed, e.g. i3d) |
| `image_modality` | str | Input modality (rgb) |
| `domain_pair` | str | Domain pair name (EPIC_D1_D2) |
| `num_classes` | int | Number of action classes |
| `frames_per_segment` | int | Frames sampled per segment (16) |

### Output Variables (Effects)
| Column | Type | Description |
|--------|------|-------------|
| `model_setup_time_seconds` | float | Time to build model |
| `training_time_seconds` | float | Training time |
| `evaluation_time_seconds` | float | Test evaluation time |
| `train_total_loss` | float | Final training loss |
| `valid_loss` | float | Validation loss |
| `test_loss` | float | Test loss |
| `memory_peak_mb` | float | Peak CPU memory (MB) |
| `gpu_memory_peak_mb` | float | Peak GPU memory (MB) |

## Production Configuration Notes

- **Same grid as digits_dann**: Identical hyperparameter space for consistent causal DAG
- **TF32 included**: H100/A100 default mode, represents real production baseline
- **fp32-true included**: True FP32 for clean causal comparison vs TF32
- **DAN included**: MMD-based method for complete causal comparison
- **4 batch sizes**: 4, 8, 16, 32 (video-appropriate equivalents of digits_dann's 64, 128, 256, 512)
- **Optimiser-specific LRs**: AdamW [3e-4, 1e-3], SGD [0.03, 0.1] (modern best practices)
- **5 seeds**: Sufficient for variance estimation in precision comparisons
- **Dataset recreated per seed**: `VideoDataset.get_source_target()` uses the seed internally
- **GPU cache cleared between runs**: Video models need explicit memory management
- **8 CPUs**: Video frame decoding is CPU-heavy
- **64G RAM**: Video frames need more system memory
- Results saved to `data/production/action_results_job*.csv`
- Total runtime: ~8 hours with 32 parallel H100 GPUs

## Files

- `action_dann_data_collector.py` - Main data collector
- `submit_action_jobs.sh` - SLURM submission script
- `merge_results.py` - Merge parallel results
- `utils/utils.py` - Auto-downloads annotation pkl files on first run
- `utils/download_frames.sh` - Downloads and extracts EPIC-Kitchens video frames
- `README.md` - This file
- `EXECUTE.md` - Quick-start guide
