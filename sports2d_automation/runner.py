from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import build_sports2d_config, write_config
from .environment import collect_environment_report, write_environment_report
from .models import AnalysisSettings, InputJob, JobResult, PreparedVideo
from .paths import DEFAULT_SPORTS2D, OUTPUTS_DIR
from .reports import generate_reports_for_job
from .video import create_run_dir, prepare_videos, unique_job_dir, video_size_from_metadata


LogCallback = Callable[[str], None]

MARKER_ERROR_RE = re.compile(r"marker error: RMS = ([0-9.]+), max = ([0-9.]+)")


class Sports2DRunner:
    def __init__(
        self,
        outputs_dir: Path = OUTPUTS_DIR,
        sports2d_exe: Path = DEFAULT_SPORTS2D,
        log: LogCallback | None = None,
    ) -> None:
        self.outputs_dir = outputs_dir
        self.sports2d_exe = sports2d_exe
        self.log = log or (lambda message: None)
        self.current_process: subprocess.Popen | None = None

    def run_jobs(
        self,
        jobs: list[InputJob],
        settings: AnalysisSettings,
        cancel_event: threading.Event | None = None,
    ) -> list[JobResult]:
        results: list[JobResult] = []
        for job in jobs:
            if cancel_event and cancel_event.is_set():
                break
            results.append(self.run_job(job, settings, cancel_event=cancel_event))
        return results

    def run_job(
        self,
        job: InputJob,
        settings: AnalysisSettings,
        cancel_event: threading.Event | None = None,
    ) -> JobResult:
        started_at = datetime.now()
        job_output_dir = unique_job_dir(self.outputs_dir, job.folder)
        output_dir = create_run_dir(self.outputs_dir, job.folder, started_at=started_at)
        work_dir = output_dir / "_work"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        config_path = output_dir / "run_config.toml"
        environment_path = output_dir / "environment_report.json"
        status_path = output_dir / "run_status.json"
        video_metadata_path = output_dir / "video_metadata.json"

        self.log(f"开始作业：{job.name}")
        _write_status(
            status_path,
            job=job,
            status="running",
            started_at=started_at,
            output_dir=output_dir,
        )
        _write_latest_run(job_output_dir, output_dir, started_at)
        environment_report = collect_environment_report()
        write_environment_report(environment_path, environment_report)

        prepared_videos = prepare_videos(job.videos, work_dir, self.log)
        _write_video_metadata(video_metadata_path, prepared_videos)
        effective_settings = _settings_with_video_size(settings, prepared_videos)

        config = build_sports2d_config(
            video_dir=work_dir,
            video_names=[video.work_path.name for video in prepared_videos],
            result_dir=output_dir,
            settings=effective_settings,
        )
        write_config(config_path, config)
        self.log(f"Sports2D 配置已写入：{config_path}")

        return_code = self._run_sports2d(config_path, log_path, cancel_event)
        html_reports: list[Path] = []
        excel_reports: list[Path] = []
        if return_code == 0:
            html_reports, excel_reports = generate_reports_for_job(
                output_dir,
                config,
                environment_report,
                self.log,
            )
            self.log(f"作业完成：{job.name}")
            status = "success"
        else:
            self.log(f"作业失败或被取消：{job.name}，退出码 {return_code}")
            status = "canceled" if cancel_event and cancel_event.is_set() else "failed"
        _write_status(
            status_path,
            job=job,
            status=status,
            started_at=started_at,
            completed_at=datetime.now(),
            output_dir=output_dir,
            return_code=return_code,
            html_reports=html_reports,
            excel_reports=excel_reports,
        )

        return JobResult(
            input_job=job,
            output_dir=output_dir,
            job_output_dir=job_output_dir,
            config_path=config_path,
            log_path=log_path,
            environment_path=environment_path,
            status_path=status_path,
            prepared_videos=prepared_videos,
            return_code=return_code,
            html_reports=html_reports,
            excel_reports=excel_reports,
        )

    def _run_sports2d(
        self,
        config_path: Path,
        log_path: Path,
        cancel_event: threading.Event | None,
    ) -> int:
        if not self.sports2d_exe.exists():
            raise FileNotFoundError(f"找不到 Sports2D 可执行文件：{self.sports2d_exe}")
        command = [str(self.sports2d_exe), "--config", str(config_path)]
        self.log("运行命令：" + " ".join(command))
        warned_marker_error = False
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(config_path.parent),
            )
            assert self.current_process.stdout is not None
            while True:
                if cancel_event and cancel_event.is_set() and self.current_process.poll() is None:
                    self.current_process.terminate()
                    self.log("收到取消请求，正在终止 Sports2D。")
                line = self.current_process.stdout.readline()
                if line:
                    text = line.rstrip()
                    log_file.write(text + "\n")
                    log_file.flush()
                    self.log(text)
                    match = MARKER_ERROR_RE.search(text)
                    if match and not warned_marker_error:
                        rms = float(match.group(1))
                        maximum = float(match.group(2))
                        if rms > 0.25 or maximum > 0.50:
                            warned_marker_error = True
                            self.log(
                                "警告：OpenSim IK marker error 已明显偏大，"
                                f"RMS={rms:.3f} m，max={maximum:.3f} m。"
                                "本次 IK/MOT 结果可能不可信，建议启用标记增强或检查拍摄角度/参数。"
                            )
                elif self.current_process.poll() is not None:
                    break
            return_code = self.current_process.wait()
            self.current_process = None
            return return_code


def _write_video_metadata(path: Path, prepared_videos: list[PreparedVideo]) -> None:
    payload = []
    for video in prepared_videos:
        payload.append(
            {
                "source_path": str(video.source_path),
                "work_path": str(video.work_path),
                "rotation_fixed": video.rotation_fixed,
                "original_metadata": video.original_metadata,
                "prepared_metadata": video.prepared_metadata,
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _settings_with_video_size(
    settings: AnalysisSettings, prepared_videos: list[PreparedVideo]
) -> AnalysisSettings:
    if not settings.input_size_auto or not prepared_videos:
        return settings
    size = video_size_from_metadata(prepared_videos[0].prepared_metadata)
    if not size:
        return settings
    return replace(settings, input_width=size[0], input_height=size[1])


def _write_latest_run(job_output_dir: Path, run_dir: Path, started_at: datetime) -> None:
    job_output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(timespec="seconds"),
    }
    (job_output_dir / "latest_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_status(
    path: Path,
    *,
    job: InputJob,
    status: str,
    started_at: datetime,
    output_dir: Path,
    completed_at: datetime | None = None,
    return_code: int | None = None,
    html_reports: list[Path] | None = None,
    excel_reports: list[Path] | None = None,
) -> None:
    payload = {
        "job": job.name,
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "completed_at": completed_at.isoformat(timespec="seconds") if completed_at else None,
        "return_code": return_code,
        "output_dir": str(output_dir),
        "html_reports": [str(path) for path in html_reports or []],
        "excel_reports": [str(path) for path in excel_reports or []],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
