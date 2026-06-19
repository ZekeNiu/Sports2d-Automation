from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Callable

from .config import build_sports2d_config, write_config
from .environment import collect_environment_report, write_environment_report
from .models import AnalysisSettings, InputJob, JobResult, PreparedVideo
from .paths import DEFAULT_SPORTS2D, OUTPUTS_DIR
from .reports import generate_reports_for_job
from .video import prepare_videos, unique_job_dir


LogCallback = Callable[[str], None]


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
        output_dir = unique_job_dir(self.outputs_dir, job.folder)
        work_dir = output_dir / "_work"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        config_path = output_dir / "run_config.toml"
        environment_path = output_dir / "environment_report.json"
        video_metadata_path = output_dir / "video_metadata.json"

        self.log(f"开始作业：{job.name}")
        environment_report = collect_environment_report()
        write_environment_report(environment_path, environment_report)

        prepared_videos = prepare_videos(job.videos, work_dir, self.log)
        _write_video_metadata(video_metadata_path, prepared_videos)

        config = build_sports2d_config(
            video_dir=work_dir,
            video_names=[video.work_path.name for video in prepared_videos],
            result_dir=output_dir,
            settings=settings,
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
        else:
            self.log(f"作业失败或被取消：{job.name}，退出码 {return_code}")

        return JobResult(
            input_job=job,
            output_dir=output_dir,
            config_path=config_path,
            log_path=log_path,
            environment_path=environment_path,
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
