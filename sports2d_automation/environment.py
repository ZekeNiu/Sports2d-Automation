from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def collect_environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": {},
        "executables": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
        },
        "onnxruntime_providers": [],
        "torch": {"installed": False, "cuda_available": False},
        "deepsort": {
            "deep_sort_realtime": importlib.util.find_spec("deep_sort_realtime") is not None,
            "torchreid": importlib.util.find_spec("torchreid") is not None,
        },
    }
    for package in [
        "sports2d",
        "pose2sim",
        "opensim",
        "PySide6",
        "plotly",
        "openpyxl",
        "pandas",
        "onnxruntime",
        "rtmlib",
    ]:
        report["packages"][package] = _version(package)
    try:
        import onnxruntime as ort

        report["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["onnxruntime_error"] = str(exc)
    try:
        import opensim as osim

        report["opensim_import"] = True
        report["opensim_version"] = getattr(osim, "__version__", report["packages"].get("opensim"))
    except Exception as exc:
        report["opensim_import"] = False
        report["opensim_error"] = str(exc)
    try:
        import torch

        report["torch"] = {
            "installed": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception:
        pass
    return report


def write_environment_report(path: Path, report: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report or collect_environment_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def check_latest_sports2d() -> str:
    cmd = [sys.executable, "-m", "pip", "index", "versions", "sports2d"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def update_sports2d_pose2sim() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-U", "sports2d", "pose2sim"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
