#!/bin/bash
#
# Downloads and extracts EPIC-Kitchens RGB frames for domains D1 (P08),
# D2 (P01) and D3 (P22) into the layout PyKale expects:
#
#   <root>/EPIC/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/{train,test}/P{xx}/P{xx}_{yy}/frame_*.jpg
#
# Only RGB frames are downloaded (the action_dann config uses image_modality=rgb).
# If you also need optical flow, change MODALITIES below.
#
# Usage:
#   bash download_frames.sh /mnt/parscratch/users/cs1fxa/datasets/EgoAction
#
# The script is resumable — already-extracted videos are skipped.

set -euo pipefail

DATASET_ROOT="${1:?Usage: $0 <dataset_root>}"
FRAMES_DIR="$DATASET_ROOT/EPIC/EPIC_KITCHENS_2018/frames_rgb_flow"
BASE_URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow"

# Change to "rgb flow" if you need optical flow as well.
MODALITIES="rgb"

if ! command -v wget &>/dev/null; then
    echo "Error: wget is not installed." >&2
    exit 1
fi

download_and_extract() {
    local modality="$1" split="$2" participant="$3" video="$4"
    local dest="$FRAMES_DIR/$modality/$split/$participant"

    # skip if already extracted
    if [ -d "$dest/$video" ]; then
        return
    fi

    mkdir -p "$dest"
    local url="$BASE_URL/$modality/$split/$participant/$video.tar"
    local tar_file="$dest/$video.tar"

    echo "  $modality/$split/$participant/$video"
    wget -q --continue -O "$tar_file" "$url"
    tar -xf "$tar_file" -C "$dest"
    rm -f "$tar_file"
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

# --- download ---

for modality in $MODALITIES; do
    echo "=== $modality ==="

    echo "--- train ---"
    for v in $P08_TRAIN; do download_and_extract "$modality" train P08 "$v"; done
    for v in $P01_TRAIN; do download_and_extract "$modality" train P01 "$v"; done
    for v in $P22_TRAIN; do download_and_extract "$modality" train P22 "$v"; done

    echo "--- test ---"
    for v in $P08_TEST; do download_and_extract "$modality" test P08 "$v"; done
    for v in $P01_TEST; do download_and_extract "$modality" test P01 "$v"; done
    for v in $P22_TEST; do download_and_extract "$modality" test P22 "$v"; done
done

echo ""
echo "Done. Frames are at: $FRAMES_DIR"
