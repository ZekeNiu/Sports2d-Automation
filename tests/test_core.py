from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sports2d_automation.config import build_sports2d_config
from sports2d_automation.models import AnalysisSettings
from sports2d_automation.reports import angle_statistics, read_mot, write_excel_report, write_html_report
from sports2d_automation.video import ffprobe_metadata, rotation_from_metadata, safe_ascii_name, unique_job_dir


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

    def test_build_config_defaults_to_full_mode(self) -> None:
        settings = AnalysisSettings()
        config = build_sports2d_config(
            Path("C:/videos"),
            ["demo.mp4"],
            Path("C:/out"),
            settings,
        )
        self.assertTrue(config["kinematics"]["do_ik"])
        self.assertTrue(config["kinematics"]["use_augmentation"])
        self.assertFalse(config["base"]["save_img"])
        self.assertEqual(config["base"]["person_ordering_method"], "highest_likelihood")

    def test_rotation_from_side_data(self) -> None:
        metadata = {"streams": [{"side_data_list": [{"rotation": -90}]}]}
        self.assertEqual(rotation_from_metadata(metadata), -90)
        metadata = {"streams": [{"tags": {"rotate": "90"}}]}
        self.assertEqual(rotation_from_metadata(metadata), 90)

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
            self.assertIn("Sports2D 交互报告", html_path.read_text(encoding="utf-8"))

            excel_path = root / "report.xlsx"
            write_excel_report(
                excel_path,
                [(mot, df)],
                {"base": {"video_input": ["demo.mp4"]}},
                {"packages": {"sports2d": "test"}},
                root,
            )
            self.assertTrue(excel_path.exists())


if __name__ == "__main__":
    unittest.main()
