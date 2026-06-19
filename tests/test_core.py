from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sports2d_automation.config import build_sports2d_config
from sports2d_automation.gui import HELP_TEXTS
from sports2d_automation.models import AnalysisSettings
from sports2d_automation.reports import (
    angle_statistics,
    collect_quality_diagnostics,
    marker_error_summary,
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

            html_path = root / "report.html"
            write_html_report(html_path, "demo", [(mot, df)], None, None)
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Sports2D 交互报告", html_text)
            self.assertIn("2D 视频平面角", html_text)

            excel_path = root / "report.xlsx"
            write_excel_report(
                excel_path,
                [(mot, df)],
                {"base": {"video_input": ["demo.mp4"]}, "kinematics": {"do_ik": False}},
                {"packages": {"sports2d": "test"}},
                root,
            )
            self.assertTrue(excel_path.exists())

    def test_trc_y_axis_is_mapped_to_display_z(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trc = Path(tmp) / "demo_m_person00.trc"
            trc.write_text(
                "PathFileType\t4\t(X/Y/Z)\tdemo.trc\n"
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n"
                "60\t60\t1\t2\tm\n"
                "Frame#\tTime\tHip\t\t\tHead\t\t\n"
                "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n"
                "1\t0.0\t1\t2\t3\t4\t5\t6\n",
                encoding="utf-8",
            )
            data = read_trc(trc)
            self.assertIsNotNone(data)
            assert data is not None
            self.assertEqual(data["marker_frames"][0]["Hip"], [1.0, 3.0, 2.0])
            self.assertEqual(data["axis_mapping"]["display_z"], "TRC Y/vertical")

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


if __name__ == "__main__":
    unittest.main()
