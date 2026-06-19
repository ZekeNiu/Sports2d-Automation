from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Callable

from .models import InputJob, PreparedVideo
from .paths import VIDEO_EXTENSIONS


LogCallback = Callable[[str], None]


def discover_input_jobs(inputs_dir: Path) -> list[InputJob]:
    jobs: list[InputJob] = []
    if not inputs_dir.exists():
        return jobs
    for folder in sorted([p for p in inputs_dir.iterdir() if p.is_dir()]):
        videos = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
        )
        if videos:
            jobs.append(InputJob(name=folder.name, folder=folder, videos=videos))
    return jobs


def safe_ascii_name(name: str, fallback: str = "job") -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    ascii_name = re.sub(r"_+", "_", ascii_name)
    return ascii_name or fallback


def unique_job_dir(outputs_dir: Path, input_folder: Path) -> Path:
    base = safe_ascii_name(input_folder.name)
    digest = hashlib.sha1(str(input_folder.resolve()).encode("utf-8")).hexdigest()[:8]
    if base != input_folder.name:
        return outputs_dir / f"{base}_{digest}"
    return outputs_dir / base


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ffprobe_metadata(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration,codec_name,pix_fmt:"
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout or "{}")


def rotation_from_metadata(metadata: dict) -> int:
    streams = metadata.get("streams") or []
    if not streams:
        return 0
    stream = streams[0]
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"]))
        except ValueError:
            return 0
    for item in stream.get("side_data_list") or []:
        if "rotation" in item:
            try:
                return int(float(item["rotation"]))
            except ValueError:
                return 0
    return 0


def prepare_videos(
    videos: list[Path],
    work_dir: Path,
    log: LogCallback | None = None,
) -> list[PreparedVideo]:
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedVideo] = []
    for index, source in enumerate(videos, start=1):
        original_metadata = ffprobe_metadata(source)
        rotation = rotation_from_metadata(original_metadata)
        safe_stem = safe_ascii_name(source.stem, fallback=f"video_{index:02d}")
        out_path = work_dir / f"{index:02d}_{safe_stem}.mp4"
        if rotation:
            if log:
                log(f"检测到视频方向元数据 rotation={rotation}: {source.name}，正在生成物理旋转副本。")
            _transcode_with_autorotation(source, out_path)
            rotation_fixed = True
        else:
            # Keep Sports2D/OpenCV on ASCII paths even if the original path is already readable.
            if source.suffix.lower() == ".mp4" and is_ascii_path(source):
                shutil.copy2(source, out_path)
            else:
                _transcode_no_rotation(source, out_path)
            rotation_fixed = False
        prepared_metadata = ffprobe_metadata(out_path)
        prepared.append(
            PreparedVideo(
                source_path=source,
                work_path=out_path,
                original_metadata=original_metadata,
                prepared_metadata=prepared_metadata,
                rotation_fixed=rotation_fixed,
            )
        )
    return prepared


def convert_video_for_browser(source: Path, target: Path, log: LogCallback | None = None) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return target
    except subprocess.CalledProcessError as exc:
        if log:
            log(f"浏览器兼容视频转码失败：{exc.stderr.strip()}")
        return None


def _transcode_with_autorotation(source: Path, target: Path) -> None:
    # ffmpeg applies display rotation by default; metadata is cleared on the new stream.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-metadata:s:v:0",
        "rotate=0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _transcode_no_rotation(source: Path, target: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-metadata:s:v:0",
        "rotate=0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
