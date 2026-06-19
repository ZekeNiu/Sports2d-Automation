from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from .config import build_sports2d_config, config_preview
from .environment import collect_environment_report
from .models import AnalysisSettings
from .parsing import parse_float_list, parse_str_list
from .paths import INPUTS_DIR, OUTPUTS_DIR
from .runner import Sports2DRunner
from .video import discover_input_jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sports2D 自动化命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="列出 Inputs 下可分析的视频作业")
    subparsers.add_parser("check-env", help="输出环境检查 JSON")

    preview_parser = subparsers.add_parser("preview-config", help="预览默认 Sports2D TOML 配置")
    _add_common_args(preview_parser)

    run_parser = subparsers.add_parser("run", help="运行一个或多个输入作业")
    _add_common_args(run_parser)
    run_parser.add_argument("--job", action="append", help="按输入文件夹名称选择；可重复。默认运行全部。")

    args = parser.parse_args(argv)
    if args.command == "list":
        for job in discover_input_jobs(INPUTS_DIR):
            print(f"{job.name}\t{len(job.videos)} video(s)")
        return 0
    if args.command == "check-env":
        print(json.dumps(collect_environment_report(), ensure_ascii=False, indent=2))
        return 0

    settings = _settings_from_args(args)
    if args.command == "preview-config":
        print(
            config_preview(
                build_sports2d_config(
                    video_dir=INPUTS_DIR,
                    video_names=["example.mp4"],
                    result_dir=OUTPUTS_DIR / "preview",
                    settings=settings,
                )
            )
        )
        return 0

    jobs = discover_input_jobs(INPUTS_DIR)
    if args.job:
        wanted = set(args.job)
        jobs = [job for job in jobs if job.name in wanted]
    if not jobs:
        print("没有找到可运行的输入作业。")
        return 1

    runner = Sports2DRunner(log=_safe_print)
    results = runner.run_jobs(jobs, settings, cancel_event=threading.Event())
    return 0 if all(result.return_code == 0 for result in results) else 2


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--height", type=float, default=1.70, help="受试者身高，米。")
    parser.add_argument("--mass", default="70", help="体重，kg；多人用逗号分隔。")
    parser.add_argument("--persons", default="1", help="检测人数，整数或 all。")
    parser.add_argument("--visible-side", default="auto", help="可见侧，逗号分隔。")
    parser.add_argument("--start", type=float, default=None, help="开始时间，秒。")
    parser.add_argument("--end", type=float, default=None, help="结束时间，秒。")
    parser.add_argument("--quick", action="store_true", help="快速检查：关闭米制转换、C3D 和 IK。")
    parser.add_argument("--ik", action="store_true", help="开启 OpenSim IK，并默认开启标记增强。")
    parser.add_argument("--no-augmentation", action="store_true", help="专家选项：开启 IK 时仍关闭标记增强。")
    parser.add_argument("--feet-on-floor", action="store_true", help="动作中双脚始终贴地时启用脚贴地修正。")
    parser.add_argument("--save-images", action="store_true", help="保存逐帧图片。")
    parser.add_argument("--device", default="auto", help="计算设备：auto/cpu/cuda/mps/rocm。")
    parser.add_argument("--backend", default="auto", help="推理后端：auto/openvino/onnxruntime/opencv。")
    parser.add_argument("--mode", default="balanced", help="检测模式：lightweight/balanced/performance。")


def _settings_from_args(args: argparse.Namespace) -> AnalysisSettings:
    settings = AnalysisSettings()
    settings.first_person_height = args.height
    settings.default_height = args.height
    settings.participant_mass = parse_float_list(args.mass, [70.0])
    settings.nb_persons_to_detect = args.persons
    settings.visible_side = parse_str_list(args.visible_side, ["auto"])
    settings.device = args.device
    settings.backend = args.backend
    settings.mode = args.mode
    settings.save_img = bool(args.save_images)
    settings.do_ik = bool(args.ik)
    settings.use_augmentation = True
    settings.feet_on_floor = bool(args.feet_on_floor)
    if args.no_augmentation:
        settings.use_augmentation = False
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            raise SystemExit("--start 和 --end 必须同时提供。")
        settings.use_time_range = True
        settings.start_time = args.start
        settings.end_time = args.end
    if args.quick:
        settings.do_ik = False
        settings.use_augmentation = False
        settings.to_meters = False
        settings.make_c3d = False
    return settings


def _safe_print(message: str) -> None:
    text = str(message)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
