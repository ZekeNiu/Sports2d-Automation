from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sports2d_automation.config import build_sports2d_config
from sports2d_automation.gui import HELP_TEXTS
from sports2d_automation.models import AnalysisSettings
from sports2d_automation.reports import (
    angle_statistics,
    collect_quality_diagnostics,
    marker_error_summary,
    measure_metadata,
    read_mot,
    read_trc,
    write_excel_report,
    write_html_report,
)
from sports2d_automation.video import (
    create_run_dir,
    ffprobe_metadata,
    rotation_from_metadata,
    safe_ascii_name,
    unique_job_dir,
    video_size_from_metadata,
)


def write_minimal_trc(path: Path) -> None:
    markers = [
        "RHip",
        "RKnee",
        "RAnkle",
        "RBigToe",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LHip",
        "LKnee",
        "LAnkle",
        "LBigToe",
        "LShoulder",
        "LElbow",
        "LWrist",
    ]
    frames = [
        {
            "time": 0.0,
            "RHip": (0.0, 0.0, 0.0),
            "RKnee": (0.0, -1.0, 0.0),
            "RAnkle": (0.0, -2.0, 0.0),
            "RBigToe": (1.0, -2.0, 0.0),
            "RShoulder": (0.0, 1.0, 0.0),
            "RElbow": (0.0, 0.0, 0.0),
            "RWrist": (0.0, -1.0, 0.0),
            "LHip": (2.0, 0.0, 0.0),
            "LKnee": (2.0, -1.0, 0.0),
            "LAnkle": (2.0, -2.0, 0.0),
            "LBigToe": (3.0, -2.0, 0.0),
            "LShoulder": (2.0, 1.0, 0.0),
            "LElbow": (2.0, 0.0, 0.0),
            "LWrist": (2.0, -1.0, 0.0),
        },
        {
            "time": 1.0,
            "RHip": (0.0, 0.0, 0.0),
            "RKnee": (0.0, -1.0, 0.0),
            "RAnkle": (1.0, -1.0, 0.0),
            "RBigToe": (2.0, -1.0, 0.0),
            "RShoulder": (0.0, 1.0, 0.0),
            "RElbow": (1.0, 1.0, 0.0),
            "RWrist": (1.0, 0.0, 0.0),
            "LHip": (2.0, 0.0, 0.0),
            "LKnee": (2.0, -1.0, 0.0),
            "LAnkle": (3.0, -1.0, 0.0),
            "LBigToe": (4.0, -1.0, 0.0),
            "LShoulder": (2.0, 1.0, 0.0),
            "LElbow": (3.0, 1.0, 0.0),
            "LWrist": (3.0, 0.0, 0.0),
        },
    ]
    marker_header = "Frame#\tTime\t" + "\t\t\t".join(markers) + "\t\t\t\n"
    coord_header = "\t\t" + "\t".join(
        label for index, _ in enumerate(markers, start=1) for label in (f"X{index}", f"Y{index}", f"Z{index}")
    )
    rows = []
    for frame_index, frame in enumerate(frames, start=1):
        values = [str(frame_index), f"{frame['time']:.3f}"]
        for marker in markers:
            values.extend(f"{value:.3f}" for value in frame[marker])
        rows.append("\t".join(values))
    path.write_text(
        "PathFileType\t4\t(X/Y/Z)\ttest.trc\n"
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        f"1\t1\t{len(frames)}\t{len(markers)}\tpx\t1\t1\t{len(frames)}\n"
        + marker_header
        + coord_header
        + "\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


class CoreTests(unittest.TestCase):
    def test_safe_ascii_name(self) -> None:
        self.assertEqual(safe_ascii_name("20260619Powerclean"), "20260619Powerclean")
        self.assertTrue(safe_ascii_name("深蹲 测试"))
        self.assertNotIn(" ", safe_ascii_name("Power clean test"))

    def test_unique_job_dir_is_stable_for_cleaned_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "Outputs"
            source = Path(tmp) / "Inputs" / "深蹲 测试"
            first = unique_job_dir(outputs, source)
            first.mkdir(parents=True)
            second = unique_job_dir(outputs, source)
            self.assertEqual(first, second)
            self.assertTrue(first.name.isascii())

    def test_create_run_dir_is_isolated_under_ascii_job_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "Outputs"
            source = Path(tmp) / "Inputs" / "深蹲 测试"
            first = create_run_dir(outputs, source)
            first.mkdir(parents=True)
            second = create_run_dir(outputs, source)
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent.name, "runs")
            self.assertTrue(first.parent.parent.name.isascii())

    def test_build_config_defaults_to_recommended_mode(self) -> None:
        settings = AnalysisSettings()
        config = build_sports2d_config(
            Path("C:/videos"),
            ["demo.mp4"],
            Path("C:/out"),
            settings,
        )
        self.assertFalse(config["kinematics"]["do_ik"])
        self.assertTrue(config["kinematics"]["use_augmentation"])
        self.assertFalse(config["kinematics"]["feet_on_floor"])
        self.assertFalse(config["base"]["save_img"])
        self.assertEqual(config["base"]["person_ordering_method"], "highest_likelihood")

    def test_rotation_and_size_from_metadata(self) -> None:
        metadata = {"streams": [{"side_data_list": [{"rotation": -90}]}]}
        self.assertEqual(rotation_from_metadata(metadata), -90)
        metadata = {"streams": [{"tags": {"rotate": "90"}}]}
        self.assertEqual(rotation_from_metadata(metadata), 90)
        metadata = {"streams": [{"width": 720, "height": 1280}]}
        self.assertEqual(video_size_from_metadata(metadata), (720, 1280))

    def test_local_rotation_probe_if_sample_exists(self) -> None:
        sample = Path("Inputs/20260619OHsquat/1.mov")
        if not sample.exists():
            self.skipTest("local sample video not present")
        metadata = ffprobe_metadata(sample)
        self.assertEqual(rotation_from_metadata(metadata), -90)

    def test_mot_reading_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mot = root / "demo_angles_person00.mot"
            mot.write_text(
                "name demo\n"
                "datacolumns 3\n"
                "datarows 3\n"
                "endheader\n"
                "time\tRight knee\tLeft knee\n"
                "0.0\t10\t20\n"
                "0.5\t30\t15\n"
                "1.0\t25\t35\n",
                encoding="utf-8",
            )
            df = read_mot(mot)
            self.assertEqual(list(df.columns), ["time", "Right knee", "Left knee"])
            stats = angle_statistics(df)
            self.assertEqual(set(stats["angle"]), {"Right knee", "Left knee"})
            self.assertNotIn("rom_note", stats.columns)
            trc = root / "demo_px_person00.trc"
            write_minimal_trc(trc)
            trc_df = read_trc(trc)
            self.assertIn("RKnee_X", trc_df.columns)
            self.assertIn("RBigToe_Y", trc_df.columns)

            ik_mot = root / "demo_ik.mot"
            ik_df = df.assign(
                hip_flexion_r=[5, 25, 15],
                pelvis_tx=[0.0, 0.1, 0.2],
                knee_angle_r_beta=[1, 2, 3],
                custom_unknown_rotation=[0.0, 1.0, 2.0],
            )[["time", "hip_flexion_r", "pelvis_tx", "knee_angle_r_beta", "custom_unknown_rotation"]]
            html_path = root / "report.html"
            quality = {
                "status": "warn",
                "warnings": ["测试质量提示"],
                "angle_notes": [],
                "marker_error_logs": [],
                "run_log_insights": {"sports2d_seen_from": "front", "configured_visible_side": "right"},
            }
            write_html_report(html_path, "demo", [(mot, df), (ik_mot, ik_df)], None, None, quality)
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("<title>Sports2D 运动学分析报告</title>", html_text)
            self.assertIn("<h1>Sports2D 运动学分析报告</h1>", html_text)
            self.assertNotIn("demo - Sports2D 运动学报告", html_text)
            self.assertIn("重点关节指标", html_text)
            self.assertIn("qualityModal", html_text)
            self.assertIn("metricModal", html_text)
            self.assertIn("metric-info", html_text)
            self.assertIn("关节活动角 (deg)", html_text)
            self.assertIn("右侧膝关节屈曲角（2D）", html_text)
            self.assertIn("0°位/中立位", html_text)
            self.assertIn('"rom": 90.0', html_text)
            self.assertNotIn("Right knee", html_text)
            self.assertIn("活动范围/数值范围", html_text)
            self.assertNotIn("<th>Metric</th>", html_text)
            self.assertNotIn("<th>动作含义</th>", html_text)
            self.assertNotIn("<th>Unit</th>", html_text)
            self.assertNotIn("ROM (deg) / Range", html_text)
            self.assertIn("OpenSim IK 关节活动角（高级）", html_text)
            self.assertIn("右侧髋关节屈曲/伸展角（OpenSim IK）", html_text)
            self.assertIn("正值：髋关节屈曲；负值：髋关节伸展", html_text)
            self.assertIn("高级诊断附录", html_text)
            self.assertIn("骨盆前后平移（辅助数据，非关节活动度）", html_text)
            self.assertIn("未分类 OpenSim 旋转坐标（高级诊断）", html_text)
            self.assertIn('"is_plottable": false', html_text)
            self.assertNotIn("OpenSim 模型坐标", html_text)
            self.assertNotIn("selected.size === 0 ||", html_text)
            self.assertIn("正面或背面视角", html_text)
            self.assertNotIn("角度定义与动作含义", html_text)
            self.assertIn("Sports2D 2D 平面活动角", html_text)
            self.assertNotIn("markerPlot", html_text)
            self.assertNotIn("三维标记视图", html_text)

            excel_path = root / "report.xlsx"
            write_excel_report(
                excel_path,
                [(mot, df)],
                {"base": {"video_input": ["demo.mp4"]}, "kinematics": {"do_ik": False}},
                {"packages": {"sports2d": "test"}},
                root,
            )
            self.assertTrue(excel_path.exists())

    def test_html_report_falls_back_when_px_trc_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mot = root / "demo_angles_person00.mot"
            mot.write_text(
                "time\tRight knee\n"
                "0.0\t10\n"
                "1.0\t40\n",
                encoding="utf-8",
            )
            html_path = root / "fallback.html"
            write_html_report(html_path, "demo", [(mot, read_mot(mot))], None, None, {})
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Sports2D 2D 原始平面角（未标准化）", html_text)
            self.assertIn("缺少对应 px TRC，无法可靠给出 0°中立位", html_text)
            self.assertIn("Right knee", html_text)

    def test_measure_metadata_explains_2d_and_ik_angles(self) -> None:
        sports2d = {"kind_short": "Sports2D 2D 平面活动角", "kind": "Sports2D 2D 平面活动角", "kind_note": ""}
        knee = measure_metadata("Right knee", sports2d)
        self.assertIn("膝关节屈曲", knee["movement_label"])
        self.assertIn("视频平面", knee["description"] + knee["interpretation"])
        self.assertIn("未标准化", knee["neutral_definition"])
        ik = {"kind_short": "OpenSim IK 关节活动角（高级）", "kind": "OpenSim IK 关节活动角（高级）", "kind_note": ""}
        hip = measure_metadata("hip_flexion_r", ik)
        self.assertIn("右侧髋关节屈曲/伸展角", hip["movement_label"])
        self.assertEqual(hip["unit"], "deg")
        self.assertTrue(hip["is_plottable"])
        self.assertIn("marker error", hip["interpretation"])
        self.assertIn("0°对应 OpenSim 模型髋关节", hip["neutral_definition"])
        self.assertIn("正值：髋关节屈曲；负值：髋关节伸展", hip["direction_definition"])
        self.assertTrue(hip["is_primary_rom_metric"])
        ankle = measure_metadata("ankle_angle_r", ik)
        self.assertIn("正值：踝关节背屈；负值：踝关节跖屈", ankle["direction_definition"])
        pelvis_rotation = measure_metadata("pelvis_rotation", ik)
        self.assertIn("正值：骨盆向左旋转；负值：骨盆向右旋转", pelvis_rotation["direction_definition"])
        pelvis_tx = measure_metadata("pelvis_tx", ik)
        self.assertEqual(pelvis_tx["unit"], "m")
        self.assertFalse(pelvis_tx["is_angle"])
        self.assertTrue(pelvis_tx["is_auxiliary"])
        self.assertFalse(pelvis_tx["is_plottable"])
        self.assertFalse(pelvis_tx["is_primary_rom_metric"])
        knee_beta = measure_metadata("knee_angle_r_beta", ik)
        self.assertTrue(knee_beta["is_auxiliary"])
        self.assertFalse(knee_beta["is_plottable"])
        unknown = measure_metadata("custom_unknown_rotation", ik)
        self.assertTrue(unknown["is_auxiliary"])
        self.assertFalse(unknown["is_plottable"])

    def test_report_statistics_sort_and_filter_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "ankle_angle_r": [1.0, 2.0],
                "hip_flexion_l": [3.0, 8.0],
                "pelvis_tx": [0.0, 0.1],
                "neck_flexion": [2.0, 5.0],
                "knee_angle_l": [4.0, 9.0],
                "hip_adduction_l": [1.0, 6.0],
            }
        )
        kind = {"kind_short": "OpenSim IK 关节活动角（高级）", "kind": "OpenSim IK 关节活动角（高级）", "kind_note": ""}
        stats = angle_statistics(df, kind, report_details=True)
        self.assertEqual(
            list(stats["angle"]),
            ["neck_flexion", "pelvis_tx", "hip_flexion_l", "hip_adduction_l", "knee_angle_l", "ankle_angle_r"],
        )
        self.assertFalse(bool(stats.loc[stats["angle"] == "pelvis_tx", "is_plottable"].iloc[0]))
        self.assertEqual(stats.loc[stats["angle"] == "pelvis_tx", "range_label"].iloc[0], "Range")
        self.assertEqual(stats.loc[stats["angle"] == "hip_flexion_l", "range_label"].iloc[0], "ROM")

    def test_marker_error_quality_diagnostics(self) -> None:
        text = "\n".join(
            [
                "Frame 1 marker error: RMS = 6.7006, max = 30.6992",
                "Frame 2 marker error: RMS = 13.3665, max = 61.3658",
            ]
        )
        summary = marker_error_summary(text)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["frames"], 2)
        self.assertGreater(summary["rms_max"], 13)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "run.log").write_text(
                text + "\nCamera horizon: 30.97°\n- Person 0: Seen from the front.\n",
                encoding="utf-8",
            )
            config = {
                "base": {"visible_side": ["front"]},
                "kinematics": {"do_ik": True, "use_augmentation": False},
            }
            quality = collect_quality_diagnostics(output_dir, config)
            self.assertEqual(quality["status"], "fail")
            self.assertTrue(any("未启用标记增强" in warning for warning in quality["warnings"]))
            self.assertEqual(quality["run_log_insights"]["camera_horizon_deg"], 30.97)

    def test_gui_help_texts_cover_key_parameters(self) -> None:
        required = {
            "preset",
            "height",
            "input_size_auto",
            "input_width",
            "do_ik",
            "augmentation",
            "feet_on_floor",
            "floor_angle",
            "cutoff",
        }
        self.assertTrue(required.issubset(HELP_TEXTS))
        self.assertGreater(len(HELP_TEXTS), 50)
        for text in HELP_TEXTS.values():
            self.assertIn("用途：", text)
            self.assertIn("默认建议：", text)
            self.assertIn("选项说明：", text)
            self.assertIn("何时修改：", text)
            self.assertIn("设置不当的风险：", text)


if __name__ == "__main__":
    unittest.main()
