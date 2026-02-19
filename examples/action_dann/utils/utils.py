import logging
import os
import pathlib
import tempfile
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

ZIP_URL = "https://raw.githubusercontent.com/pykale/data/main/videos/video_test_data.zip"
EPIC_PREFIX = "video_test_data/EPIC/"


def ensure_dataset(dataset_root: str) -> None:
    """Download and extract the EPIC video test data if ``dataset_root`` does not exist

    The data is fetched from the pykale/data repository on GitHub (~584 KB zip).
    Only the ``EPIC/`` subdirectory is extracted so that the resulting layout is

        <dataset_root>/EPIC/EPIC_KITCHENS_2018/annotations/labels_train_test/*.pkl

    Args:
        dataset_root: Path where the dataset should reside.
    """
    root = pathlib.Path(dataset_root)
    if root.exists():
        logger.info("Dataset root already exists at %s — skipping download.", root)
        return

    logger.info("Downloading video_test_data.zip from %s ...", ZIP_URL)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    try:
        os.close(tmp_fd)
        urllib.request.urlretrieve(ZIP_URL, tmp_path)
        logger.info("Download complete. Extracting EPIC data to %s ...", root)

        with zipfile.ZipFile(tmp_path, "r") as zf:
            for member in zf.infolist():
                if not member.filename.startswith(EPIC_PREFIX):
                    continue
                # Strip the leading "video_test_data/" so files land under dataset_root directly.
                rel_path = member.filename[len("video_test_data/"):]
                if not rel_path:
                    continue
                target = root / rel_path
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

        logger.info("Dataset ready at %s", root)
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
