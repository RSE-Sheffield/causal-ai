#!/bin/bash
#SBATCH --job-name=epic-frames
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/download_frames_%j.out
#SBATCH --error=logs/download_frames_%j.err
#
# Downloads and extracts EPIC-Kitchens RGB frames for domains D1 (P08),
# D2 (P01) and D3 (P22) into the layout PyKale expects:
#
#   <root>/EPIC/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/{train,test}/P{xx}/P{xx}_{yy}/frame_*.jpg
#
# Usage:
#   sbatch download_frames_bede.sh
#
# Resumable — already-extracted videos are skipped.

set -euo pipefail

mkdir -p logs 2>/dev/null || true

DATASET_ROOT="/nobackup/projects/bddur53/cs1fxa/causal-ai/datasets/EgoAction"
FRAMES_DIR="$DATASET_ROOT/EPIC/EPIC_KITCHENS_2018/frames_rgb_flow"
BASE_URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow"

# How many videos to download at once. 8 is a good balance between
# speed and not hammering the Bristol server.
MAX_PARALLEL=8

if ! command -v wget &>/dev/null; then
    echo "Error: wget is not installed." >&2
    exit 1
fi

download_and_extract() {
    local modality="$1" split="$2" participant="$3" video="$4"
    local dest="$FRAMES_DIR/$modality/$split/$participant"

    # skip if already extracted (check for actual frames, not just the directory)
    if find "$dest/$video" -maxdepth 1 -name "frame_*.jpg" -print -quit 2>/dev/null | grep -q .; then
        echo "  [skip] $modality/$split/$participant/$video"
        return 0
    fi

    mkdir -p "$dest/$video"
    local url="$BASE_URL/$modality/$split/$participant/$video.tar"
    local tar_file="$dest/${video}_$$.tar"

    echo "  [get]  $modality/$split/$participant/$video"
    if ! wget -q -O "$tar_file" "$url"; then
        echo "  [FAIL] $modality/$split/$participant/$video — download failed" >&2
        rm -f "$tar_file"
        rmdir "$dest/$video" 2>/dev/null || true
        return 1
    fi

    tar -xf "$tar_file" -C "$dest/$video"
    rm -f "$tar_file"
    echo "  [done] $modality/$split/$participant/$video"
}

# --- video lists (from MM-SADA domain adaptation splits) ---

# D1 = P08
P08_TRAIN="P08_01 P08_02 P08_03 P08_04 P08_05 P08_06 P08_07 P08_08
           P08_11 P08_12 P08_13 P08_18 P08_19 P08_20 P08_21 P08_22
           P08_23 P08_24 P08_25 P08_26 P08_27 P08_28"
P08_TEST="P08_09 P08_10 P08_14 P08_15 P08_16 P08_17"

# D2 = P01
P01_TRAIN="P01_01 P01_02 P01_03 P01_04 P01_05 P01_06 P01_07 P01_08
           P01_09 P01_10 P01_16 P01_17 P01_18 P01_19"
P01_TEST="P01_11 P01_12 P01_13 P01_14 P01_15"

# D3 = P22
P22_TRAIN="P22_05 P22_06 P22_07 P22_08 P22_09 P22_10 P22_11 P22_12
           P22_13 P22_14 P22_15 P22_16 P22_17"
P22_TEST="P22_01 P22_02 P22_03 P22_04"

# --- build flat task list ---

TASKS=()
for v in $P08_TRAIN; do TASKS+=("rgb train P08 $v"); done
for v in $P01_TRAIN; do TASKS+=("rgb train P01 $v"); done
for v in $P22_TRAIN; do TASKS+=("rgb train P22 $v"); done
for v in $P08_TEST;  do TASKS+=("rgb test  P08 $v"); done
for v in $P01_TEST;  do TASKS+=("rgb test  P01 $v"); done
for v in $P22_TEST;  do TASKS+=("rgb test  P22 $v"); done

echo "=== EPIC-Kitchens frame download ==="
echo "Destination: $FRAMES_DIR"
echo "Videos: ${#TASKS[@]} total, $MAX_PARALLEL in parallel"
echo ""

# --- download in parallel (batches of MAX_PARALLEL) ---

batch=0
for task in "${TASKS[@]}"; do
    download_and_extract $task &
    batch=$((batch + 1))

    if ((batch >= MAX_PARALLEL)); then
        wait
        batch=0
    fi
done
wait

echo ""
echo "Done. Frames are at: $FRAMES_DIR"
