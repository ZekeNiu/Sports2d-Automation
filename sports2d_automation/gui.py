from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config import build_sports2d_config, config_preview
from .environment import collect_environment_report, update_sports2d_pose2sim
from .models import AnalysisSettings, InputJob, JobResult
from .parsing import parse_float_list, parse_str_list
from .paths import INPUTS_DIR, OUTPUTS_DIR
from .runner import Sports2DRunner
from .video import discover_input_jobs
from .parameter_help import HELP_TEXTS


PRESET_RECOMMENDED = "推荐新手模式"
PRESET_FULL = "完整 OpenSim 模式"
PRESET_EXPERT = "专家模式"

class RunWorker(QObject):
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, jobs: list[InputJob], settings: AnalysisSettings) -> None:
        super().__init__()
        self.jobs = jobs
        self.settings = settings
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            runner = Sports2DRunner(log=self.log.emit)
            results = runner.run_jobs(self.jobs, self.settings, self.cancel_event)
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()


class UpdateWorker(QObject):
    log = Signal(str)
    finished = Signal(int)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            process = update_sports2d_pose2sim()
            assert process.stdout is not None
            for line in process.stdout:
                self.log.emit(line.rstrip())
            self.finished.emit(process.wait())
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sports2D 自动化分析工具")
        self.resize(1280, 860)
        self.jobs: list[InputJob] = []
        self.worker: RunWorker | None = None
        self.worker_thread: QThread | None = None
        self.update_worker: UpdateWorker | None = None
        self.update_thread: QThread | None = None
        self._build_ui()
        self.refresh_jobs()
        self.refresh_preview()

    def _build_ui(self) -> None:
        root = QWidget()
        main_layout = QVBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_job_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.setSizes([360, 900])
        main_layout.addWidget(splitter, stretch=1)

        main_layout.addWidget(self._build_action_panel())
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(170)
        main_layout.addWidget(self.log_text)
        self.setCentralWidget(root)

    def _build_job_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("输入视频作业")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)
        layout.addWidget(QLabel("从 Inputs 下的子文件夹扫描视频。勾选一个或多个作业后运行。"))
        self.job_list = QListWidget()
        layout.addWidget(self.job_list, stretch=1)
        row = QHBoxLayout()
        self.refresh_button = QPushButton("刷新")
        self.select_all_button = QPushButton("全选")
        self.clear_selection_button = QPushButton("清空")
        self.refresh_button.clicked.connect(self.refresh_jobs)
        self.select_all_button.clicked.connect(lambda: self._set_all_jobs(Qt.Checked))
        self.clear_selection_button.clicked.connect(lambda: self._set_all_jobs(Qt.Unchecked))
        row.addWidget(self.refresh_button)
        row.addWidget(self.select_all_button)
        row.addWidget(self.clear_selection_button)
        layout.addLayout(row)
        return panel

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("分析预设"))
        self.preset_combo = _combo([PRESET_RECOMMENDED, PRESET_FULL, PRESET_EXPERT])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(self._help_button("preset"))
        preset_row.addStretch(1)
        layout.addLayout(preset_row)
        self.preset_hint = QLabel(
            "推荐新手模式默认不运行 OpenSim IK；先确认 2D 骨架和角度稳定，再切换完整 OpenSim 模式。"
        )
        self.preset_hint.setWordWrap(True)
        self.preset_hint.setStyleSheet("color:#5f6b7a;")
        layout.addWidget(self.preset_hint)

        self.tabs = QTabWidget()
        self.base_tab_index = self.tabs.addTab(self._base_tab(), "基础信息")
        self.output_tab_index = self.tabs.addTab(self._output_tab(), "输出")
        self.pose_tab_index = self.tabs.addTab(self._pose_tab(), "姿态检测")
        self.calibration_tab_index = self.tabs.addTab(self._calibration_tab(), "尺度/标定")
        self.post_tab_index = self.tabs.addTab(self._post_tab(), "后处理")
        self.ik_tab_index = self.tabs.addTab(self._ik_tab(), "逆运动学")
        self.toml_tab_index = self.tabs.addTab(self._toml_tab(), "TOML 预览")
        layout.addWidget(self.tabs, stretch=1)
        self.preset_combo.setCurrentText(PRESET_RECOMMENDED)
        self.apply_preset(PRESET_RECOMMENDED)
        return panel

    def _base_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.height_spin = _double_spin(0.5, 2.5, 1.70, 0.01, " m")
        self.mass_edit = QLineEdit("70")
        self.persons_combo = _combo(["1", "2", "all"], editable=True)
        self.order_combo = _combo(
            [
                "highest_likelihood",
                "largest_size",
                "smallest_size",
                "greatest_displacement",
                "least_displacement",
                "first_detected",
                "last_detected",
                "on_click",
            ]
        )
        self.visible_side_edit = QLineEdit("auto")
        self.time_range_check = QCheckBox("只分析指定时间范围")
        self.start_spin = _double_spin(0, 99999, 0, 0.1, " s")
        self.end_spin = _double_spin(0, 99999, 1, 0.1, " s")
        self.slowmo_spin = _double_spin(0.01, 100, 1.0, 0.1, " x")
        self._add_row(form, "身高", self.height_spin, "height")
        self._add_row(form, "体重 kg（多人用逗号分隔）", self.mass_edit, "mass")
        self._add_row(form, "检测人数", self.persons_combo, "persons")
        self._add_row(form, "人物排序方式", self.order_combo, "order")
        self._add_row(form, "可见侧（auto/right/left/front/back/none）", self.visible_side_edit, "visible_side")
        self._add_check_row(form, self.time_range_check, "time_range")
        self._add_row(form, "开始时间", self.start_spin, "start")
        self._add_row(form, "结束时间", self.end_spin, "end")
        self._add_row(form, "慢动作倍率", self.slowmo_spin, "slowmo")
        self._connect_preview(tab)
        return tab

    def _output_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.save_video_check = QCheckBox("保存处理后视频")
        self.save_video_check.setChecked(True)
        self.save_images_check = QCheckBox("保存逐帧图片")
        self.save_pose_check = QCheckBox("保存姿态 TRC")
        self.save_pose_check.setChecked(True)
        self.save_angles_check = QCheckBox("保存角度 MOT")
        self.save_angles_check.setChecked(True)
        self.calculate_angles_check = QCheckBox("计算关节/节段角度")
        self.calculate_angles_check.setChecked(True)
        self.save_graphs_check = QCheckBox("保存 Sports2D 原生图表")
        self.save_graphs_check.setChecked(True)
        self.show_graphs_check = QCheckBox("运行时显示图表窗口")
        self.realtime_check = QCheckBox("运行时显示实时结果窗口")
        layout.addWidget(self._checkbox_help_widget(self.save_video_check, "save_video"))
        note = QLabel("HTML/Excel 报告所需的角度、MOT 和 TRC 输出默认由系统保持开启。逐帧图片和运行时弹窗默认隐藏在专家模式中。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#5f6b7a;")
        layout.addWidget(note)
        self.output_expert_group = QGroupBox("专家参数")
        expert_layout = QVBoxLayout(self.output_expert_group)
        for widget in [
            self._checkbox_help_widget(self.save_images_check, "save_images"),
            self._checkbox_help_widget(self.save_pose_check, "save_pose"),
            self._checkbox_help_widget(self.save_angles_check, "save_angles"),
            self._checkbox_help_widget(self.calculate_angles_check, "calculate_angles"),
            self._checkbox_help_widget(self.save_graphs_check, "save_graphs"),
            self._checkbox_help_widget(self.show_graphs_check, "show_graphs"),
            self._checkbox_help_widget(self.realtime_check, "realtime"),
        ]:
            expert_layout.addWidget(widget)
        layout.addWidget(self.output_expert_group)
        layout.addStretch(1)
        self._connect_preview(tab)
        return tab

    def _pose_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.pose_model_combo = _combo(["body_with_feet", "whole_body_wrist", "whole_body", "body"])
        self.mode_combo = _combo(["balanced", "lightweight", "performance"], editable=True)
        self.det_frequency_spin = _int_spin(1, 500, 4)
        self.tracking_combo = _combo(["sports2d", "deepsort"])
        self.device_combo = _combo(["auto", "cpu", "cuda", "mps", "rocm"])
        self.backend_combo = _combo(["auto", "openvino", "onnxruntime", "opencv"])
        self.input_size_auto_check = QCheckBox("自动读取普通视频宽高")
        self.input_size_auto_check.setChecked(True)
        self.input_width_spin = _int_spin(128, 4096, 1280)
        self.input_height_spin = _int_spin(128, 4096, 720)
        self.kpt_threshold_spin = _double_spin(0, 1, 0.3, 0.05)
        self.avg_threshold_spin = _double_spin(0, 1, 0.5, 0.05)
        self.kpt_number_spin = _double_spin(0, 1, 0.3, 0.05)
        self.max_distance_spin = _int_spin(0, 5000, 250)
        self.max_unseen_spin = _double_spin(0, 60, 1.0, 0.1, " s")
        self._add_row(form, "姿态模型", self.pose_model_combo, "pose_model")
        self._add_row(form, "检测模式", self.mode_combo, "mode")
        self._add_row(form, "检测频率（每 N 帧检测一次人）", self.det_frequency_spin, "det_frequency")
        self._add_check_row(form, self.input_size_auto_check, "input_size_auto")
        layout.addLayout(form)

        self.pose_expert_group = QGroupBox("专家参数")
        expert_form = QFormLayout(self.pose_expert_group)
        self._add_row(expert_form, "跟踪模式", self.tracking_combo, "tracking")
        self._add_row(expert_form, "计算设备", self.device_combo, "device")
        self._add_row(expert_form, "推理后端", self.backend_combo, "backend")
        self._add_row(expert_form, "输入宽度", self.input_width_spin, "input_width")
        self._add_row(expert_form, "输入高度", self.input_height_spin, "input_height")
        self._add_row(expert_form, "关键点置信度阈值", self.kpt_threshold_spin, "kpt_threshold")
        self._add_row(expert_form, "平均置信度阈值", self.avg_threshold_spin, "avg_threshold")
        self._add_row(expert_form, "关键点数量阈值", self.kpt_number_spin, "kpt_number")
        self._add_row(expert_form, "最大跳变距离 px", self.max_distance_spin, "max_distance")
        self._add_row(expert_form, "最大丢失时间", self.max_unseen_spin, "max_unseen")
        layout.addWidget(self.pose_expert_group)
        layout.addStretch(1)
        self._connect_preview(tab)
        return tab

    def _calibration_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.to_meters_check = QCheckBox("像素转换为米")
        self.to_meters_check.setChecked(True)
        self.make_c3d_check = QCheckBox("生成 C3D")
        self.make_c3d_check.setChecked(True)
        self.save_calib_check = QCheckBox("保存标定文件")
        self.save_calib_check.setChecked(True)
        self.floor_angle_edit = QLineEdit("auto")
        self.xy_origin_edit = QLineEdit("auto")
        self.perspective_spin = _double_spin(0, 100000, 10.0, 0.5)
        self.perspective_unit_combo = _combo(["distance_m", "f_px", "fov_deg", "fov_rad", "from_calib"])
        self.calib_file_edit = QLineEdit("")
        calib_row = QHBoxLayout()
        calib_row.addWidget(self.calib_file_edit)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse_calib)
        calib_row.addWidget(browse)
        self._add_check_row(form, self.to_meters_check, "to_meters")
        self._add_check_row(form, self.make_c3d_check, "make_c3d")
        self._add_check_row(form, self.save_calib_check, "save_calib")
        layout.addLayout(form)
        self.calibration_expert_group = QGroupBox("专家参数")
        expert_form = QFormLayout(self.calibration_expert_group)
        self._add_row(expert_form, "地面角（auto/from_calib/数值）", self.floor_angle_edit, "floor_angle")
        self._add_row(expert_form, "XY 原点（auto 或 x,y）", self.xy_origin_edit, "xy_origin")
        self._add_row(expert_form, "透视值", self.perspective_spin, "perspective")
        self._add_row(expert_form, "透视单位", self.perspective_unit_combo, "perspective_unit")
        self._add_row(expert_form, "标定文件", calib_row, "calib_file")
        layout.addWidget(self.calibration_expert_group)
        layout.addStretch(1)
        self._connect_preview(tab)
        return tab

    def _post_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.interpolate_check = QCheckBox("插值缺失数据")
        self.interpolate_check.setChecked(True)
        self.interp_gap_spin = _int_spin(0, 10000, 100)
        self.fill_gaps_combo = _combo(["last_value", "nan", "zeros"])
        self.sections_combo = _combo(["all", "largest", "first", "last"])
        self.min_chunk_spin = _int_spin(1, 10000, 10)
        self.reject_outliers_check = QCheckBox("Hampel 去除离群值")
        self.reject_outliers_check.setChecked(True)
        self.filter_check = QCheckBox("滤波")
        self.filter_check.setChecked(True)
        self.filter_type_combo = _combo(["butterworth", "kalman", "gcv_spline", "gaussian", "median", "loess"])
        self.cutoff_spin = _double_spin(0.1, 1000, 6.0, 0.5, " Hz")
        self.filter_order_spin = _int_spin(1, 12, 4)
        self._add_check_row(form, self.interpolate_check, "interpolate")
        self._add_row(form, "最大插值间隙（帧）", self.interp_gap_spin, "interp_gap")
        self._add_row(form, "大间隙填充方式", self.fill_gaps_combo, "fill_gaps")
        self._add_row(form, "保留片段", self.sections_combo, "sections")
        self._add_row(form, "最小有效片段帧数", self.min_chunk_spin, "min_chunk")
        self._add_check_row(form, self.reject_outliers_check, "reject_outliers")
        self._add_check_row(form, self.filter_check, "filter")
        self._add_row(form, "滤波类型", self.filter_type_combo, "filter_type")
        self._add_row(form, "Butterworth 截止频率", self.cutoff_spin, "cutoff")
        self._add_row(form, "Butterworth 阶数", self.filter_order_spin, "filter_order")
        self._connect_preview(tab)
        return tab

    def _ik_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.do_ik_check = QCheckBox("运行 OpenSim 逆运动学")
        self.do_ik_check.setChecked(False)
        self.augmentation_check = QCheckBox("运行标记增强")
        self.augmentation_check.setChecked(True)
        self.feet_on_floor_check = QCheckBox("动作中双脚始终贴地时启用修正")
        self.feet_on_floor_check.setChecked(False)
        self.simple_model_check = QCheckBox("使用简单 OpenSim 模型")
        self.symmetry_check = QCheckBox("左右对称")
        self.symmetry_check.setChecked(True)
        self.default_height_spin = _double_spin(0.5, 2.5, 1.70, 0.01, " m")
        self.large_angle_spin = _double_spin(0, 180, 135, 1)
        self.trimmed_spin = _double_spin(0, 1, 0.5, 0.05)
        self.osim_setup_edit = QLineEdit("../OpenSim_setup")
        self.remove_scaling_check = QCheckBox("删除单个 scaling setup 临时文件")
        self.remove_scaling_check.setChecked(True)
        self.remove_ik_check = QCheckBox("删除单个 IK setup 临时文件")
        self.remove_ik_check.setChecked(True)
        self._add_check_row(form, self.do_ik_check, "do_ik")
        self._add_check_row(form, self.augmentation_check, "augmentation")
        self._add_check_row(form, self.feet_on_floor_check, "feet_on_floor")
        self._add_check_row(form, self.simple_model_check, "simple_model")
        self._add_check_row(form, self.symmetry_check, "symmetry")
        self._add_row(form, "默认身高", self.default_height_spin, "default_height")
        layout.addLayout(form)

        self.ik_expert_group = QGroupBox("专家参数")
        expert_form = QFormLayout(self.ik_expert_group)
        self._add_row(expert_form, "大髋/膝角阈值", self.large_angle_spin, "large_angle")
        self._add_row(expert_form, "极值裁剪比例", self.trimmed_spin, "trimmed")
        self._add_row(expert_form, "OpenSim setup 路径", self.osim_setup_edit, "osim_setup")
        self._add_check_row(expert_form, self.remove_scaling_check, "remove_scaling")
        self._add_check_row(expert_form, self.remove_ik_check, "remove_ik")
        layout.addWidget(self.ik_expert_group)
        layout.addStretch(1)
        self.do_ik_check.stateChanged.connect(self._sync_ik_safety)
        self.augmentation_check.stateChanged.connect(self._sync_ik_safety)
        self._connect_preview(tab)
        return tab

    def _toml_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        refresh = QPushButton("刷新预览")
        refresh.clicked.connect(self.refresh_preview)
        row.addWidget(refresh)
        row.addStretch(1)
        layout.addLayout(row)
        note = QLabel("说明：普通视频运行时会自动读取真实宽高；预览中的 input_size 仅是 webcam/占位值。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#5f6b7a;")
        layout.addWidget(note)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        layout.addWidget(self.preview_text)
        return tab

    def _build_action_panel(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        self.run_button = QPushButton("运行所选分析")
        self.cancel_button = QPushButton("取消当前任务")
        self.cancel_button.setEnabled(False)
        self.check_env_button = QPushButton("检查环境")
        self.update_button = QPushButton("一键更新 Sports2D/Pose2Sim")
        self.open_outputs_button = QPushButton("打开输出目录")
        self.run_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.check_env_button.clicked.connect(self.check_environment)
        self.update_button.clicked.connect(self.start_update)
        self.open_outputs_button.clicked.connect(lambda: os.startfile(OUTPUTS_DIR))
        layout.addWidget(self.run_button, 0, 0)
        layout.addWidget(self.cancel_button, 0, 1)
        layout.addWidget(self.check_env_button, 0, 2)
        layout.addWidget(self.update_button, 0, 3)
        layout.addWidget(self.open_outputs_button, 0, 4)
        return panel

    @Slot()
    def refresh_jobs(self) -> None:
        self.jobs = discover_input_jobs(INPUTS_DIR)
        self.job_list.clear()
        for job in self.jobs:
            item = QListWidgetItem(f"{job.name}  ({len(job.videos)} 个视频)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, job)
            self.job_list.addItem(item)
        self.log(f"已扫描 {len(self.jobs)} 个输入作业。")
        self.refresh_preview()

    @Slot()
    def refresh_preview(self) -> None:
        if not hasattr(self, "preview_text"):
            return
        try:
            settings = self.collect_settings()
            config = build_sports2d_config(
                video_dir=INPUTS_DIR,
                video_names=["example.mp4"],
                result_dir=OUTPUTS_DIR / "preview",
                settings=settings,
            )
            self.preview_text.setPlainText(config_preview(config))
        except Exception as exc:
            self.preview_text.setPlainText(f"参数错误：{exc}")

    @Slot()
    def start_analysis(self) -> None:
        jobs = self.selected_jobs()
        if not jobs:
            QMessageBox.warning(self, "没有选择作业", "请先勾选至少一个输入视频作业。")
            return
        try:
            settings = self.collect_settings()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = RunWorker(jobs, settings)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()
        self.log("已启动分析任务。")

    @Slot()
    def cancel_analysis(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.log("已请求取消，正在等待当前子进程退出。")

    @Slot(object)
    def analysis_finished(self, results: list[JobResult]) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        success = sum(1 for result in results if result.return_code == 0)
        self.log(f"分析结束：成功 {success}/{len(results)}。")
        for result in results:
            self.log(f"输出目录：{result.output_dir}")
            for report in result.html_reports + result.excel_reports:
                self.log(f"报告：{report}")

    @Slot(str)
    def analysis_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.log(f"分析失败：{message}")
        QMessageBox.critical(self, "分析失败", message)

    @Slot()
    def check_environment(self) -> None:
        report = collect_environment_report()
        self.log("环境检查结果：")
        self.log(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["deepsort"]["deep_sort_realtime"]:
            self.log("提示：DeepSort 依赖未安装；选择 deepsort 跟踪模式前需要额外安装 deep_sort_realtime/torchreid。")

    @Slot()
    def start_update(self) -> None:
        self.update_button.setEnabled(False)
        worker = UpdateWorker()
        thread = QThread()
        self.update_worker = worker
        self.update_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self.log)
        worker.finished.connect(lambda code: self._update_finished(code, worker, thread))
        worker.failed.connect(lambda msg: self._update_failed(msg, worker, thread))
        thread.start()
        self.log("开始更新 sports2d 和 pose2sim。")

    def _update_finished(self, code: int, worker: UpdateWorker, thread: QThread) -> None:
        self.update_button.setEnabled(True)
        self.log(f"更新命令结束，退出码 {code}。")
        thread.quit()
        worker.deleteLater()
        self.update_worker = None

    def _update_failed(self, message: str, worker: UpdateWorker, thread: QThread) -> None:
        self.update_button.setEnabled(True)
        self.log(f"更新失败：{message}")
        thread.quit()
        worker.deleteLater()
        self.update_worker = None

    def selected_jobs(self) -> list[InputJob]:
        selected: list[InputJob] = []
        for row in range(self.job_list.count()):
            item = self.job_list.item(row)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def collect_settings(self) -> AnalysisSettings:
        settings = AnalysisSettings()
        settings.preset = self.preset_combo.currentText()
        settings.first_person_height = self.height_spin.value()
        settings.default_height = self.default_height_spin.value()
        settings.participant_mass = parse_float_list(self.mass_edit.text(), [70.0])
        settings.nb_persons_to_detect = self.persons_combo.currentText()
        settings.person_ordering_method = self.order_combo.currentText()
        settings.visible_side = parse_str_list(self.visible_side_edit.text(), ["auto"])
        settings.use_time_range = self.time_range_check.isChecked()
        settings.start_time = self.start_spin.value()
        settings.end_time = self.end_spin.value()
        settings.show_realtime_results = self.realtime_check.isChecked()
        settings.save_vid = self.save_video_check.isChecked()
        settings.save_img = self.save_images_check.isChecked()
        settings.save_pose = self.save_pose_check.isChecked()
        settings.calculate_angles = self.calculate_angles_check.isChecked()
        settings.save_angles = self.save_angles_check.isChecked()
        settings.save_graphs = self.save_graphs_check.isChecked()
        settings.show_graphs = self.show_graphs_check.isChecked()
        settings.slowmo_factor = self.slowmo_spin.value()
        settings.pose_model = self.pose_model_combo.currentText()
        settings.mode = self.mode_combo.currentText()
        settings.det_frequency = self.det_frequency_spin.value()
        settings.backend = self.backend_combo.currentText()
        settings.device = self.device_combo.currentText()
        settings.tracking_mode = self.tracking_combo.currentText()
        settings.input_size_auto = self.input_size_auto_check.isChecked()
        settings.input_width = self.input_width_spin.value()
        settings.input_height = self.input_height_spin.value()
        settings.keypoint_likelihood_threshold = self.kpt_threshold_spin.value()
        settings.average_likelihood_threshold = self.avg_threshold_spin.value()
        settings.keypoint_number_threshold = self.kpt_number_spin.value()
        settings.max_distance = self.max_distance_spin.value()
        settings.max_unseen_time = self.max_unseen_spin.value()
        settings.to_meters = self.to_meters_check.isChecked()
        settings.make_c3d = self.make_c3d_check.isChecked()
        settings.save_calib = self.save_calib_check.isChecked()
        settings.floor_angle = self.floor_angle_edit.text().strip() or "auto"
        settings.xy_origin = parse_str_list(self.xy_origin_edit.text(), ["auto"])
        if len(settings.xy_origin) == 2:
            try:
                settings.xy_origin = [float(settings.xy_origin[0]), float(settings.xy_origin[1])]
            except ValueError:
                pass
        settings.perspective_value = self.perspective_spin.value()
        settings.perspective_unit = self.perspective_unit_combo.currentText()
        settings.calib_file = self.calib_file_edit.text().strip()
        settings.interpolate = self.interpolate_check.isChecked()
        settings.interp_gap_smaller_than = self.interp_gap_spin.value()
        settings.fill_large_gaps_with = self.fill_gaps_combo.currentText()
        settings.sections_to_keep = self.sections_combo.currentText()
        settings.min_chunk_size = self.min_chunk_spin.value()
        settings.reject_outliers = self.reject_outliers_check.isChecked()
        settings.filter = self.filter_check.isChecked()
        settings.filter_type = self.filter_type_combo.currentText()
        settings.butterworth_cutoff = self.cutoff_spin.value()
        settings.butterworth_order = self.filter_order_spin.value()
        settings.do_ik = self.do_ik_check.isChecked()
        settings.use_augmentation = self.augmentation_check.isChecked()
        if settings.do_ik and not settings.use_augmentation and settings.preset != PRESET_EXPERT:
            settings.use_augmentation = True
            self.augmentation_check.setChecked(True)
        settings.feet_on_floor = self.feet_on_floor_check.isChecked()
        settings.use_simple_model = self.simple_model_check.isChecked()
        settings.right_left_symmetry = self.symmetry_check.isChecked()
        settings.large_hip_knee_angles = self.large_angle_spin.value()
        settings.trimmed_extrema_percent = self.trimmed_spin.value()
        settings.osim_setup_path = self.osim_setup_edit.text().strip() or "../OpenSim_setup"
        settings.remove_individual_scaling_setup = self.remove_scaling_check.isChecked()
        settings.remove_individual_ik_setup = self.remove_ik_check.isChecked()
        settings.time_range()
        return settings

    def _browse_calib(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择标定文件", str(INPUTS_DIR), "TOML (*.toml);;所有文件 (*.*)")
        if path:
            self.calib_file_edit.setText(Path(path).as_posix())

    def _set_all_jobs(self, state: Qt.CheckState) -> None:
        for row in range(self.job_list.count()):
            self.job_list.item(row).setCheckState(state)

    @Slot(str)
    def apply_preset(self, preset: str) -> None:
        if not hasattr(self, "do_ik_check"):
            return
        expert = preset == PRESET_EXPERT
        self._set_expert_visible(expert)
        if preset == PRESET_RECOMMENDED:
            self.preset_hint.setText(
                "推荐新手模式：生成 2D 角度、处理后视频、TRC/MOT、HTML 和 Excel；不默认运行 OpenSim IK。"
            )
            self.do_ik_check.setChecked(False)
            self.augmentation_check.setChecked(True)
            self.feet_on_floor_check.setChecked(False)
            self.pose_model_combo.setCurrentText("body_with_feet")
            self.mode_combo.setCurrentText("balanced")
            self.det_frequency_spin.setValue(4)
            self.input_size_auto_check.setChecked(True)
        elif preset == PRESET_FULL:
            self.preset_hint.setText(
                "完整 OpenSim 模式：在推荐输出基础上运行 IK，并强制默认启用标记增强；仍建议先检查质量诊断。"
            )
            self.do_ik_check.setChecked(True)
            self.augmentation_check.setChecked(True)
            self.feet_on_floor_check.setChecked(False)
            self.pose_model_combo.setCurrentText("body_with_feet")
            self.mode_combo.setCurrentText("balanced")
            self.det_frequency_spin.setValue(4)
            self.input_size_auto_check.setChecked(True)
        else:
            self.preset_hint.setText(
                "专家模式：显示所有底层参数，并允许关闭标记增强；报告会对高风险组合做醒目标记。"
            )
        self._sync_ik_safety()
        self.refresh_preview()

    def _sync_ik_safety(self, *_args: object) -> None:
        if not hasattr(self, "augmentation_check"):
            return
        expert = self.preset_combo.currentText() == PRESET_EXPERT if hasattr(self, "preset_combo") else False
        if self.do_ik_check.isChecked() and not self.augmentation_check.isChecked() and not expert:
            self.augmentation_check.blockSignals(True)
            self.augmentation_check.setChecked(True)
            self.augmentation_check.blockSignals(False)
        self.augmentation_check.setEnabled(expert or not self.do_ik_check.isChecked() or self.augmentation_check.isChecked())

    def _set_expert_visible(self, visible: bool) -> None:
        for group_name in [
            "output_expert_group",
            "pose_expert_group",
            "calibration_expert_group",
            "ik_expert_group",
        ]:
            group = getattr(self, group_name, None)
            if group is not None:
                group.setVisible(visible)
        if hasattr(self, "tabs"):
            self.tabs.setTabVisible(self.post_tab_index, visible)
            self.tabs.setTabVisible(self.toml_tab_index, visible)

    def _add_row(
        self,
        form: QFormLayout,
        label: str,
        field: QWidget | QLayout,
        help_key: str,
    ) -> None:
        form.addRow(label, self._field_with_help(field, help_key))

    def _add_check_row(self, form: QFormLayout, checkbox: QCheckBox, help_key: str) -> None:
        form.addRow("", self._checkbox_help_widget(checkbox, help_key))

    def _field_with_help(self, field: QWidget | QLayout, help_key: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        if isinstance(field, QLayout):
            layout.addLayout(field)
        else:
            layout.addWidget(field)
        layout.addWidget(self._help_button(help_key))
        return container

    def _checkbox_help_widget(self, checkbox: QCheckBox, help_key: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(checkbox)
        layout.addStretch(1)
        layout.addWidget(self._help_button(help_key))
        return container

    def _help_button(self, help_key: str) -> QToolButton:
        text = HELP_TEXTS[help_key]
        button = QToolButton()
        button.setText("?")
        button.setToolTip(text)
        button.setFixedWidth(24)
        button.clicked.connect(lambda _checked=False, key=help_key: self._show_help(key))
        return button

    def _show_help(self, help_key: str) -> None:
        QMessageBox.information(self, "参数说明", HELP_TEXTS[help_key])

    def _connect_preview(self, widget: QWidget) -> None:
        children = []
        for widget_type in [QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit]:
            children.extend(widget.findChildren(widget_type))
        for child in children:
            if isinstance(child, QCheckBox):
                child.stateChanged.connect(self.refresh_preview)
            elif isinstance(child, QComboBox):
                child.currentTextChanged.connect(self.refresh_preview)
            elif isinstance(child, (QSpinBox, QDoubleSpinBox)):
                child.valueChanged.connect(self.refresh_preview)
            elif isinstance(child, QLineEdit):
                child.textChanged.connect(self.refresh_preview)

    def log(self, message: str) -> None:
        self.log_text.append(str(message))
        self.log_text.ensureCursorVisible()


def _combo(items: list[str], editable: bool = False) -> QComboBox:
    combo = QComboBox()
    combo.addItems(items)
    combo.setEditable(editable)
    return combo


def _double_spin(minimum: float, maximum: float, value: float, step: float, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(3)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
