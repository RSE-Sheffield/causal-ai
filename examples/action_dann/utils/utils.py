"""Helpers for fetching EPIC-Kitchens domain-adaptation annotation data."""

import logging
import os
import pathlib
import tempfile
import urllib.request
import zipfile

import pandas as pd

logger = logging.getLogger(__name__)

ZIP_URL = "https://raw.githubusercontent.com/pykale/data/main/videos/video_test_data.zip"

# The pykale test zip only ships D1 annotations. D2 and D3 come from the
# MM-SADA splits repo (https://github.com/jonmun/MM-SADA_Domain_Adaptation_Splits).
_MMSADA = "https://raw.githubusercontent.com/jonmun/MM-SADA_Domain_Adaptation_Splits/master"
_EXTRA_PKLS = {
    "epic_D2_train.pkl": f"{_MMSADA}/D2_train.pkl",
    "epic_D2_test.pkl": f"{_MMSADA}/D2_test.pkl",
    "epic_D3_train.pkl": f"{_MMSADA}/D3_train.pkl",
    "epic_D3_test.pkl": f"{_MMSADA}/D3_test.pkl",
}

# MM-SADA pkl columns are ordered:
#   uid, video_id, narration, start_timestamp, stop_timestamp,
#   start_frame, stop_frame, participant_id, verb, verb_class
#
# PyKale's EPIC class indexes rows positionally (via .values), expecting
# participant_id at [1] and video_id at [2]. We reorder to suit.
_PYKALE_COLS = [
    "uid", "participant_id", "video_id", "narration",
    "start_timestamp", "stop_timestamp", "start_frame",
    "stop_frame", "verb", "verb_class",
]


def _fetch_mmsada_pkl(url, dest):
    """Download a single MM-SADA pkl and reorder its columns for PyKale."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pkl")
    try:
        os.close(tmp_fd)
        urllib.request.urlretrieve(url, tmp_path)
        df = pd.read_pickle(tmp_path)
        df = df[_PYKALE_COLS]
        df.columns = range(len(df.columns))
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(str(dest))
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


def ensure_dataset(dataset_root):
    """Download EPIC-Kitchens annotation data if *dataset_root* doesn't exist.

    Grabs the pykale test-data zip (D1 annotations + directory scaffold) then
    fetches D2/D3 splits from the MM-SADA repo so that domain pairs like
    D1->D2 work out of the box.
    """
    root = pathlib.Path(dataset_root)
    if root.exists():
        logger.info("Dataset root already exists at %s, skipping.", root)
        return

    # -- grab the pykale test-data zip (contains D1 + directory structure) --
    logger.info("Downloading video_test_data.zip ...")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    try:
        os.close(tmp_fd)
        urllib.request.urlretrieve(ZIP_URL, tmp_path)
        logger.info("Extracting EPIC data to %s ...", root)

        prefix = "video_test_data/EPIC/"
        strip = "video_test_data/"
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for member in zf.infolist():
                if not member.filename.startswith(prefix):
                    continue
                rel = member.filename[len(strip):]
                if not rel:
                    continue
                target = root / rel
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    # -- fetch D2/D3 annotations from MM-SADA --
    ann_dir = root / "EPIC" / "EPIC_KITCHENS_2018" / "annotations" / "labels_train_test"
    for name, url in _EXTRA_PKLS.items():
        dest = ann_dir / name
        if dest.exists():
            continue
        logger.info("Downloading %s ...", name)
        _fetch_mmsada_pkl(url, dest)

    logger.info("Dataset ready at %s", root)
