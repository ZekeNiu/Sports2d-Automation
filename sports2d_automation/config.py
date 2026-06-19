from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from .models import AnalysisSettings


def build_sports2d_config(
    video_dir: Path,
    video_names: list[str],
    result_dir: Path,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    """Build a Sports2D-compatible TOML dictionary."""

    return {
        "base": {
            "video_input": video_names,
            "nb_persons_to_detect": _person_count(settings.nb_persons_to_detect),
            "person_ordering_method": settings.person_ordering_method,
            "first_person_height": float(settings.first_person_height),
            "visible_side": settings.visible_side,
            "participant_mass": settings.participant_mass,
            "load_trc_px": "",
            "compare": False,
            "time_range": settings.time_range(),
            "video_dir": _as_posix(video_dir),
            "webcam_id": 0,
            "input_size": [int(settings.input_width), int(settings.input_height)],
            "show_realtime_results": bool(settings.show_realtime_results),
            "save_vid": bool(settings.save_vid),
            "save_img": bool(settings.save_img),
            "save_pose": bool(settings.save_pose),
            "calculate_angles": bool(settings.calculate_angles),
            "save_angles": bool(settings.save_angles),
            "result_dir": _as_posix(result_dir),
        },
        "pose": {
            "slowmo_factor": float(settings.slowmo_factor),
            "pose_model": settings.pose_model,
            "mode": settings.mode,
            "det_frequency": int(settings.det_frequency),
            "device": settings.device,
            "backend": settings.backend,
            "tracking_mode": settings.tracking_mode,
            "deepsort_params": (
                "{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, "
                "'max_cosine_distance':0.3, 'nn_budget':200, "
                "'max_iou_distance':0.8, 'embedder_gpu': True, 'embedder':'torchreid'}"
            ),
            "keypoint_likelihood_threshold": float(settings.keypoint_likelihood_threshold),
            "average_likelihood_threshold": float(settings.average_likelihood_threshold),
            "keypoint_number_threshold": float(settings.keypoint_number_threshold),
            "max_distance": int(settings.max_distance),
            "max_unseen_time": float(settings.max_unseen_time),
        },
        "px_to_meters_conversion": {
            "to_meters": bool(settings.to_meters),
            "make_c3d": bool(settings.make_c3d),
            "save_calib": bool(settings.save_calib),
            "floor_angle": _maybe_float(settings.floor_angle),
            "xy_origin": settings.xy_origin,
            "perspective_value": float(settings.perspective_value),
            "perspective_unit": settings.perspective_unit,
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            "calib_file": settings.calib_file,
        },
        "angles": {
            "display_angle_values_on": settings.display_angle_values_on,
            "fontSize": float(settings.font_size),
            "joint_angles": settings.joint_angles,
            "segment_angles": settings.segment_angles,
            "correct_segment_angles_with_floor_angle": bool(
                settings.correct_segment_angles_with_floor_angle
            ),
        },
        "post-processing": {
            "interpolate": bool(settings.interpolate),
            "interp_gap_smaller_than": int(settings.interp_gap_smaller_than),
            "fill_large_gaps_with": settings.fill_large_gaps_with,
            "sections_to_keep": settings.sections_to_keep,
            "min_chunk_size": int(settings.min_chunk_size),
            "reject_outliers": bool(settings.reject_outliers),
            "filter": bool(settings.filter),
            "show_graphs": bool(settings.show_graphs),
            "save_graphs": bool(settings.save_graphs),
            "filter_type": settings.filter_type,
            "butterworth": {
                "cut_off_frequency": float(settings.butterworth_cutoff),
                "order": int(settings.butterworth_order),
            },
            "kalman": {"trust_ratio": 500.0, "smooth": True},
            "gcv_spline": {"gcv_cut_off_frequency": "auto", "gcv_smoothing_factor": 1.0},
            "loess": {"nb_values_used": 5},
            "gaussian": {"sigma_kernel": 1},
            "median": {"kernel_size": 3},
            "butterworth_on_speed": {
                "butterspeed_order": int(settings.butterworth_order),
                "butterspeed_cut_off_frequency": float(settings.butterworth_cutoff),
                "order": int(settings.butterworth_order),
                "cut_off_frequency": float(settings.butterworth_cutoff),
            },
        },
        "kinematics": {
            "do_ik": bool(settings.do_ik),
            "use_augmentation": bool(settings.use_augmentation),
            "feet_on_floor": bool(settings.feet_on_floor),
            "use_simple_model": bool(settings.use_simple_model),
            "participant_mass": settings.participant_mass,
            "right_left_symmetry": bool(settings.right_left_symmetry),
            "default_height": float(settings.default_height),
            "large_hip_knee_angles": float(settings.large_hip_knee_angles),
            "trimmed_extrema_percent": float(settings.trimmed_extrema_percent),
            "remove_individual_scaling_setup": bool(settings.remove_individual_scaling_setup),
            "remove_individual_ik_setup": bool(settings.remove_individual_ik_setup),
            "osim_setup_path": settings.osim_setup_path,
        },
        "logging": {"use_custom_logging": False},
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(config), encoding="utf-8")


def config_preview(config: dict[str, Any]) -> str:
    return toml.dumps(config)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, path))
        else:
            flat[path] = value
    return flat


def _as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def _person_count(value: str) -> int | str:
    text = str(value).strip().lower()
    if text == "all":
        return "all"
    return int(float(text))


def _maybe_float(value: str) -> str | float:
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text
