from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "Inputs"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
DEFAULT_ENV_DIR = Path(r"D:\Application\Anaconda\envs\sports3d")
DEFAULT_PYTHON = DEFAULT_ENV_DIR / "python.exe"
DEFAULT_SPORTS2D = DEFAULT_ENV_DIR / "Scripts" / "sports2d.exe"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
}
