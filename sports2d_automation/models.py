from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class InputJob:
    name: str
    folder: Path
    videos: list[Path]


@dataclass
class AnalysisSettings:
    # Base
    nb_persons_to_detect: str = "1"
    person_ordering_method: str = "highest_likelihood"
    first_person_height: float = 1.70
    visible_side: list[str] = field(default_factory=lambda: ["auto"])
    participant_mass: list[float] = field(default_factory=lambda: [70.0])
    use_time_range: bool = False
    start_time: float = 0.0
    end_time: float = 1.0

    # Output
    show_realtime_results: bool = False
    save_vid: bool = True
    save_img: bool = False
    save_pose: bool = True
    calculate_angles: bool = True
    save_angles: bool = True
    save_graphs: bool = True
    show_graphs: bool = False

    # Pose
    slowmo_factor: float = 1.0
    pose_model: str = "body_with_feet"
    mode: str = "balanced"
    det_frequency: int = 4
    backend: str = "auto"
    device: str = "auto"
    tracking_mode: str = "sports2d"
    input_width: int = 1280
    input_height: int = 720
    keypoint_likelihood_threshold: float = 0.3
    average_likelihood_threshold: float = 0.5
    keypoint_number_threshold: float = 0.3
    max_distance: int = 250
    max_unseen_time: float = 1.0

    # Conversion / calibration
    to_meters: bool = True
    make_c3d: bool = True
    floor_angle: str = "auto"
    xy_origin: list[str | float] = field(default_factory=lambda: ["auto"])
    perspective_value: float = 10.0
    perspective_unit: str = "distance_m"
    calib_file: str = ""
    save_calib: bool = True

    # Angles
    display_angle_values_on: list[str] = field(default_factory=lambda: ["body", "list"])
    joint_angles: list[str] = field(
        default_factory=lambda: [
            "Right ankle",
            "Left ankle",
            "Right knee",
            "Left knee",
            "Right hip",
            "Left hip",
            "Right shoulder",
            "Left shoulder",
            "Right elbow",
            "Left elbow",
            "Right wrist",
            "Left wrist",
        ]
    )
    segment_angles: list[str] = field(
        default_factory=lambda: [
            "Right foot",
            "Left foot",
            "Right shank",
            "Left shank",
            "Right thigh",
            "Left thigh",
            "Pelvis",
            "Trunk",
            "Shoulders",
            "Head",
            "Right arm",
            "Left arm",
            "Right forearm",
            "Left forearm",
        ]
    )
    font_size: float = 0.3
    correct_segment_angles_with_floor_angle: bool = True

    # Post-processing
    interpolate: bool = True
    interp_gap_smaller_than: int = 100
    fill_large_gaps_with: str = "last_value"
    sections_to_keep: str = "all"
    min_chunk_size: int = 10
    reject_outliers: bool = True
    filter: bool = True
    filter_type: str = "butterworth"
    butterworth_cutoff: float = 6.0
    butterworth_order: int = 4

    # Kinematics
    do_ik: bool = True
    use_augmentation: bool = True
    feet_on_floor: bool = True
    use_simple_model: bool = False
    right_left_symmetry: bool = True
    default_height: float = 1.70
    large_hip_knee_angles: float = 135.0
    trimmed_extrema_percent: float = 0.5
    remove_individual_scaling_setup: bool = True
    remove_individual_ik_setup: bool = True
    osim_setup_path: str = "../OpenSim_setup"

    def time_range(self) -> list[float]:
        if not self.use_time_range:
            return []
        if self.end_time <= self.start_time:
            raise ValueError("结束时间必须大于开始时间。")
        return [float(self.start_time), float(self.end_time)]


@dataclass(frozen=True)
class PreparedVideo:
    source_path: Path
    work_path: Path
    original_metadata: dict
    prepared_metadata: dict
    rotation_fixed: bool


@dataclass(frozen=True)
class JobResult:
    input_job: InputJob
    output_dir: Path
    config_path: Path
    log_path: Path
    environment_path: Path
    prepared_videos: list[PreparedVideo]
    return_code: int
    html_reports: list[Path]
    excel_reports: list[Path]
