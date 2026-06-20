from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .config import flatten_dict
from .video import convert_video_for_browser, rotation_from_metadata, video_size_from_metadata


LogCallback = Callable[[str], None]

MARKER_ERROR_RE = re.compile(r"marker error: RMS = ([0-9.]+), max = ([0-9.]+)")
HORIZON_RE = re.compile(r"Camera horizon:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", re.I)
HEIGHT_RE = re.compile(r"person height of [0-9.]+ in meters.*?of ([0-9.]+) in pixels", re.I)
SEEN_FROM_RE = re.compile(r"Seen from the ([A-Za-z ]+)", re.I)

IMPORTANT_KEYWORDS = (
    "knee",
    "hip",
    "ankle",
    "shoulder",
    "elbow",
    "wrist",
    "pelvis",
    "trunk",
    "膝",
    "髋",
    "踝",
    "肩",
    "肘",
    "腕",
    "骨盆",
    "躯干",
)

REPORT_TITLE = "Sports2D 运动学分析报告"
ROM_NOTE = (
    "本报告中的 ROM（range of motion）表示当前分析时间范围内该指标的最大值减最小值，"
    "单位为度。它是本次视频片段中实际观测到的角度变化范围，不是临床或解剖学意义上的"
    "最大关节活动度，也不代表受试者的生理活动上限。"
)
STANDARD_ROM_NOTE = (
    "本报告中的 ROM 使用标准化后的关节活动角 display_value 计算，表示当前分析时间范围内"
    "该活动角的最大值减最小值，单位为度。它反映本视频片段中观测到的活动范围，"
    "不是临床最大关节活动度。"
)
AUXILIARY_RANGE_NOTE = (
    "该指标不是关节活动度。Range 表示该辅助量在当前分析时间范围内的最大值减最小值，"
    "仅用于排查 OpenSim 模型或骨盆/躯干平移变化，不应解释为关节 ROM。"
)

BODY_RANKS = {
    "head": 10,
    "neck": 10,
    "trunk": 20,
    "spine": 20,
    "lumbar": 20,
    "pelvis": 30,
    "shoulder": 40,
    "arm": 40,
    "elbow": 50,
    "forearm": 50,
    "wrist": 60,
    "hip": 70,
    "thigh": 70,
    "knee": 80,
    "shank": 80,
    "ankle": 90,
    "subtalar": 92,
    "mtp": 94,
    "foot": 100,
}


def read_mot(mot_path: Path) -> pd.DataFrame:
    lines = mot_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_end = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = idx
            break
    if header_end is None:
        for idx, line in enumerate(lines):
            if line.strip().lower().startswith("time"):
                header_end = idx - 1
                break
    skiprows = 0 if header_end is None else header_end + 1
    return pd.read_csv(mot_path, sep="\t", skiprows=skiprows)


def read_trc(trc_path: Path) -> pd.DataFrame:
    """Read an OpenSim/Sports2D TRC file into columns like RHip_X, RHip_Y, RHip_Z."""
    lines = trc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 6:
        return pd.DataFrame()
    marker_cells = lines[3].split("\t")
    markers = [cell.strip() for cell in marker_cells[2:] if cell.strip()]
    if not markers:
        return pd.DataFrame()
    try:
        data = pd.read_csv(trc_path, sep="\t", skiprows=5, header=None, engine="python")
    except Exception:
        return pd.DataFrame()
    data = data.dropna(axis=1, how="all")
    coord_count = max(0, data.shape[1] - 2)
    marker_count = min(len(markers), coord_count // 3)
    if marker_count <= 0:
        return pd.DataFrame()
    markers = markers[:marker_count]
    columns = ["frame", "time"] + [
        f"{marker}_{axis}" for marker in markers for axis in ("X", "Y", "Z")
    ]
    data = data.iloc[:, : len(columns)].copy()
    data.columns = columns[: data.shape[1]]
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def generate_reports_for_job(
    output_dir: Path,
    config: dict[str, Any],
    environment_report: dict[str, Any],
    log: LogCallback | None = None,
) -> tuple[list[Path], list[Path]]:
    quality = collect_quality_diagnostics(output_dir, config)
    sports_dirs = sorted(
        [
            p
            for p in output_dir.iterdir()
            if p.is_dir() and (p.name.endswith("_Sports2D") or list(p.glob("*.mot")))
        ]
    )
    html_reports: list[Path] = []
    all_motions: list[tuple[Path, pd.DataFrame]] = []
    for sports_dir in sports_dirs:
        motions = _load_motion_files(sports_dir)
        all_motions.extend(motions)
        if not motions:
            continue
        report_path = sports_dir / "reports" / f"{sports_dir.name}_interactive.html"
        video = _prepare_report_video(sports_dir, log)
        write_html_report(report_path, sports_dir.name, motions, video, None, quality)
        html_reports.append(report_path)
        if log:
            log(f"HTML 交互报告已生成：{report_path}")

    excel_reports: list[Path] = []
    if all_motions:
        excel_path = output_dir / "analysis_report.xlsx"
        write_excel_report(excel_path, all_motions, config, environment_report, output_dir, quality)
        excel_reports.append(excel_path)
        if log:
            log(f"Excel 报告已生成：{excel_path}")
    return html_reports, excel_reports


def collect_quality_diagnostics(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    marker_logs: list[dict[str, Any]] = []
    insights: dict[str, Any] = {}

    kinematics = config.get("kinematics", {})
    base = config.get("base", {})
    do_ik = bool(kinematics.get("do_ik"))
    use_augmentation = bool(kinematics.get("use_augmentation"))
    if do_ik and not use_augmentation:
        warnings.append(
            "OpenSim IK 已开启，但未启用标记增强。该组合容易产生较大的 marker error，"
            "因此本报告不会把 IK 结果视为默认可信的三维运动。"
        )
    if not do_ik:
        insights["ik_status"] = "not_run"

    visible_side = [str(v).lower() for v in base.get("visible_side", [])]
    if visible_side:
        insights["configured_visible_side"] = ", ".join(visible_side)
    if any(side in {"front", "back"} for side in visible_side):
        warnings.append(
            "可见侧设置为 front/back。Sports2D 原生角度仍是视频平面角，"
            "此时不要把它解释为矢状面的髋、膝、踝屈伸角。"
        )
    elif "auto" in visible_side:
        warnings.append(
            "可见侧为 auto。请核对处理后视频和日志中的方向判断；如果左右或前后不符合实际，应手动指定可见侧。"
        )

    for log_path in _quality_log_paths(output_dir):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        marker_summary = marker_error_summary(text)
        if marker_summary:
            status = _marker_status(marker_summary)
            marker_logs.append(
                {
                    "path": str(log_path.relative_to(output_dir)),
                    "status": status,
                    **marker_summary,
                }
            )
            if status == "fail":
                warnings.append(
                    f"{log_path.name} 的 OpenSim marker error 明显异常："
                    f"最大 RMS {marker_summary['rms_max']:.3f} m，最大误差 {marker_summary['max_max']:.3f} m。"
                    "OpenSim MOT/OSIM 动作不应作为可信结果使用。"
                )
            elif status == "warn":
                warnings.append(
                    f"{log_path.name} 的 OpenSim marker error 偏大："
                    f"最大 RMS {marker_summary['rms_max']:.3f} m，最大误差 {marker_summary['max_max']:.3f} m。"
                )

        horizon = _first_float(HORIZON_RE, text)
        if horizon is not None:
            insights.setdefault("camera_horizon_deg", horizon)
            if abs(horizon) > 10:
                warnings.append(
                    f"Sports2D 自动估计的 camera horizon 为 {horizon:.2f} 度。"
                    "如果地面在真实场景中并未明显倾斜，应检查拍摄角度、地面角或标定参数。"
                )
        height_px = _first_float(HEIGHT_RE, text)
        if height_px is not None:
            insights.setdefault("person_height_px", height_px)
            if height_px < 250:
                warnings.append(
                    f"用于尺度换算的人体高度只有 {height_px:.1f}px。像素到米、TRC/C3D 和 IK 结果可能不稳定。"
                )
        seen_from = SEEN_FROM_RE.search(text)
        if seen_from:
            insights.setdefault("sports2d_seen_from", seen_from.group(1).strip())

    return {
        "status": _overall_quality_status(marker_logs, warnings),
        "warnings": _deduplicate(warnings),
        "marker_error_logs": marker_logs,
        "run_log_insights": insights,
        "video_metadata": _load_video_metadata(output_dir),
        "angle_notes": [
            "本报告的核心结果是标准化后的关节活动角。Sports2D 2D 指标优先由像素关键点重新计算，0°位用于描述视频平面内的近似中立参考，不等同于完整三维解剖测量。",
            "当摄像机近似垂直于动作平面时，膝、髋、踝等 2D 活动角可用于描述该平面内的屈伸趋势；摄像机为正面、背面或明显斜拍时，应解释为视频平面投影活动角，而不是矢状面关节角。",
            "OpenSim *_ik.mot 若存在，仅将旋转类 coordinate 整理为关节活动角展示。只有 marker error、拍摄方向、尺度和标定均通过检查时，才建议进一步用于三维解释；平移和辅助约束数据不作为关节活动度。",
        ],
    }


def marker_error_summary(text: str) -> dict[str, Any] | None:
    pairs = [(float(rms), float(maximum)) for rms, maximum in MARKER_ERROR_RE.findall(text)]
    if not pairs:
        return None
    rms_values = [pair[0] for pair in pairs]
    max_values = [pair[1] for pair in pairs]
    return {
        "frames": len(pairs),
        "rms_min": min(rms_values),
        "rms_max": max(rms_values),
        "rms_mean": sum(rms_values) / len(rms_values),
        "max_min": min(max_values),
        "max_max": max(max_values),
        "max_mean": sum(max_values) / len(max_values),
    }


def write_html_report(
    report_path: Path,
    title: str,
    motions: list[tuple[Path, pd.DataFrame]],
    video_path: Path | None,
    trc_data: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
) -> None:
    del trc_data
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plotly_js = get_plotlyjs()
    quality_payload = quality or {}
    motion_payload = [_motion_payload(path, df, quality_payload) for path, df in motions]
    figs = [_motion_figure(payload) for payload in motion_payload]
    fig_json = [json.loads(pio.to_json(fig, validate=False)) for fig in figs]
    video_rel = _relative_posix(video_path, report_path.parent) if video_path else ""
    serializable_motion_payload = [
        {key: value for key, value in payload.items() if key != "df"} for payload in motion_payload
    ]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{REPORT_TITLE}</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fa; --panel:#ffffff; --ink:#17202a; --muted:#5f6b7a; --line:#d7dee8; --accent:#0f766e; --accent-soft:#e6f4f1; --warn:#a15c07; --warn-soft:#fff4df; --bad:#b42318; --bad-soft:#fff0ee; --good:#047857; --good-soft:#e8f7ef; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei", "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--ink); line-height:1.55; }}
    header {{ padding:22px 28px 16px; border-bottom:1px solid var(--line); background:var(--panel); }}
    h1 {{ margin:0 0 6px; font-size:24px; font-weight:700; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:18px; font-weight:700; letter-spacing:0; }}
    h3 {{ margin:0 0 8px; font-size:15px; font-weight:700; letter-spacing:0; }}
    .header-row {{ max-width:1480px; margin:0 auto; }}
    main {{ max-width:1480px; margin:0 auto; padding:18px 24px 32px; display:grid; grid-template-columns:minmax(420px, 0.95fr) minmax(480px, 1.15fr); gap:16px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }}
    .full {{ grid-column:1 / -1; }}
    .muted {{ color:var(--muted); font-size:13px; }}
    .note {{ color:var(--muted); font-size:13px; margin:6px 0 0; }}
    video {{ width:100%; max-height:62vh; background:#111827; border-radius:6px; }}
    .plot {{ width:100%; height:540px; }}
    .tabs, .metric-controls {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
    .tabs button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; min-height:34px; }}
    .tabs button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
    .metric-option {{ display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); border-radius:6px; padding:6px 8px; background:#fff; font-size:13px; cursor:pointer; }}
    .metric-option input {{ margin:0; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:10px; }}
    .metric-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
    .metric-card .label {{ color:var(--muted); font-size:12px; }}
    .metric-card .value {{ font-size:22px; font-weight:700; margin:3px 0; }}
    .metric-card dl {{ display:grid; grid-template-columns:auto 1fr; gap:4px 8px; margin:8px 0 0; font-size:13px; }}
    .metric-card dt {{ color:var(--muted); }}
    .metric-card dd {{ margin:0; text-align:right; font-variant-numeric:tabular-nums; }}
    .seek-button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:3px 7px; cursor:pointer; font-size:12px; }}
    .quality {{ display:grid; grid-template-columns:190px minmax(0, 1fr); gap:14px; align-items:start; }}
    .quality-summary {{ margin-top:12px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fff; }}
    .quality-summary-title {{ display:flex; align-items:center; gap:8px; flex:0 0 auto; }}
    .quality-summary-text {{ margin:0; color:var(--muted); font-size:13px; flex:1 1 360px; }}
    .badge {{ display:inline-block; border-radius:999px; padding:5px 10px; color:#fff; background:var(--muted); font-weight:700; }}
    .badge.good {{ background:var(--good); }}
    .badge.warn {{ background:var(--warn); }}
    .badge.fail {{ background:var(--bad); }}
    .icon-button {{ display:inline-flex; align-items:center; justify-content:center; min-width:32px; min-height:32px; border:1px solid var(--line); border-radius:999px; background:#fff; color:var(--ink); font-weight:700; cursor:pointer; }}
    .icon-button:hover, .icon-button:focus-visible {{ border-color:var(--accent); outline:2px solid var(--accent-soft); outline-offset:2px; }}
    .detail-button {{ border:1px solid var(--accent); background:var(--accent-soft); color:#0f4f49; border-radius:999px; padding:6px 10px; min-height:34px; cursor:pointer; font-weight:700; }}
    .detail-button:hover, .detail-button:focus-visible {{ outline:2px solid var(--accent-soft); outline-offset:2px; }}
    .info-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }}
    .info-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
    .info-card.good {{ background:var(--good-soft); border-color:#b7e4cc; }}
    .info-card.warn {{ background:var(--warn-soft); border-color:#f0d49a; }}
    .info-card.fail {{ background:var(--bad-soft); border-color:#f4b6ad; }}
    ul {{ margin:0; padding-left:18px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child {{ text-align:left; }}
    th.metric-col, td.metric-col {{ text-align:left; min-width:190px; }}
    .metric-name {{ display:inline-flex; align-items:center; gap:6px; }}
    .metric-info {{ min-width:24px; min-height:24px; font-size:12px; }}
    .table-wrap {{ overflow-x:auto; max-width:100%; }}
    .stats-table {{ min-width:920px; }}
    .quality-table {{ min-width:720px; table-layout:fixed; }}
    .quality-table th, .quality-table td {{ word-break:break-word; overflow-wrap:anywhere; }}
    .modal-backdrop {{ position:fixed; inset:0; z-index:1000; display:flex; align-items:center; justify-content:center; padding:24px; background:rgba(15, 23, 42, 0.58); }}
    .modal-backdrop[hidden] {{ display:none; }}
    .modal-panel {{ width:min(920px, 100%); max-height:86vh; overflow:auto; border-radius:10px; background:#fff; border:1px solid var(--line); box-shadow:0 24px 60px rgba(15, 23, 42, 0.24); }}
    .modal-header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:14px; padding:16px 18px 10px; border-bottom:1px solid var(--line); }}
    .modal-content {{ padding:16px 18px 18px; }}
    .modal-content p {{ margin:0 0 10px; }}
    .modal-content dl {{ display:grid; grid-template-columns:140px minmax(0, 1fr); gap:8px 14px; margin:0; }}
    .modal-content dt {{ color:var(--muted); font-weight:700; }}
    .modal-content dd {{ margin:0; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns:1fr; padding:12px; }} .quality {{ grid-template-columns:1fr; }} .plot {{ height:460px; }} .modal-content dl {{ grid-template-columns:1fr; }} }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header>
    <div class="header-row">
      <div>
        <h1>{REPORT_TITLE}</h1>
      </div>
      {_quality_summary_html(quality_payload)}
    </div>
  </header>
  <main>
    <section>
      <h2>视频核对</h2>
      {_video_html(video_rel)}
      <p class="note">请先确认处理后视频中的 2D 骨架是否稳定、是否跟随同一名受试者、左右方向是否符合实际。若视频叠加骨架已经错误，角度曲线不应继续用于结论。</p>
    </section>
    <section>
      <h2>关节活动角曲线</h2>
      <div id="motionTabs" class="tabs"></div>
      <div id="anglePlot" class="plot"></div>
    </section>
    <section class="full">
      <h2>重点关节指标</h2>
      <p class="note">勾选需要关注的关节或节段。卡片显示最小值、最大值、活动范围 ROM 和峰值时间；点击时间按钮可同步视频到对应时刻。</p>
      <div id="metricControls" class="metric-controls"></div>
      <div id="metricCards" class="cards"></div>
    </section>
    <section class="full">
      <h2>完整统计表</h2>
      <div id="statsTable"></div>
    </section>
    <section id="auxiliarySection" class="full" hidden>
      <h2>高级诊断附录</h2>
      <p class="note">本附录列出未进入主报告的平移坐标或辅助约束数据。它们可用于排查 OpenSim 求解质量，但不属于关节活动度，也不应作为训练或康复结论的主指标。</p>
      <div id="auxiliaryTable"></div>
    </section>
  </main>
  <div id="qualityModal" class="modal-backdrop" hidden role="dialog" aria-modal="true" aria-labelledby="qualityModalTitle">
    <div class="modal-panel">
      <div class="modal-header">
        <h2 id="qualityModalTitle">质量诊断与解释边界</h2>
        <button class="icon-button" type="button" data-close-modal aria-label="关闭质量诊断">×</button>
      </div>
      <div class="modal-content">
        {_quality_detail_html(quality_payload)}
      </div>
    </div>
  </div>
  <div id="metricModal" class="modal-backdrop" hidden role="dialog" aria-modal="true" aria-labelledby="metricModalTitle">
    <div class="modal-panel">
      <div class="modal-header">
        <h2 id="metricModalTitle">指标解释</h2>
        <button class="icon-button" type="button" data-close-modal aria-label="关闭指标解释">×</button>
      </div>
      <div id="metricModalContent" class="modal-content"></div>
    </div>
  </div>
  <script>
    const motionPayload = {json.dumps(serializable_motion_payload, ensure_ascii=False)};
    const figPayload = {json.dumps(fig_json, ensure_ascii=False)};
    const video = document.getElementById('syncVideo');
    const tabs = document.getElementById('motionTabs');
    const plot = document.getElementById('anglePlot');
    const controls = document.getElementById('metricControls');
    const cards = document.getElementById('metricCards');
    const statsTable = document.getElementById('statsTable');
    const auxiliarySection = document.getElementById('auxiliarySection');
    const auxiliaryTable = document.getElementById('auxiliaryTable');
    const metricModal = document.getElementById('metricModal');
    const metricModalTitle = document.getElementById('metricModalTitle');
    const metricModalContent = document.getElementById('metricModalContent');
    const qualityModal = document.getElementById('qualityModal');
    let currentIndex = 0;

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function fmt(value, digits = 2) {{
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : 'NA';
    }}
    function selectedNames() {{
      return [...controls.querySelectorAll('input[type="checkbox"]:checked')].map(input => input.value);
    }}
    function seekTo(timeValue) {{
      const t = Number(timeValue);
      if (video && Number.isFinite(t)) {{
        video.currentTime = Math.max(0, t);
        video.pause();
      }}
    }}

    function renderMotion(index) {{
      currentIndex = index;
      Plotly.react(plot, figPayload[index].data, figPayload[index].layout, {{responsive:true}});
      [...tabs.children].forEach((button, i) => button.classList.toggle('active', i === index));
      renderMetricControls(index);
      renderMetricCards(index);
      statsTable.innerHTML = motionPayload[index].stats_html;
      const auxiliaryHtml = motionPayload[index].auxiliary_stats_html || '';
      auxiliaryTable.innerHTML = auxiliaryHtml;
      auxiliarySection.hidden = !auxiliaryHtml;
      plot.on('plotly_hover', event => {{
        const x = event.points && event.points[0] ? Number(event.points[0].x) : NaN;
        if (video && Number.isFinite(x)) video.currentTime = Math.max(0, x);
      }});
    }}

    function renderMetricControls(index) {{
      const item = motionPayload[index];
      const defaults = new Set(item.default_metrics || []);
      const plottedStats = item.stats.filter(stat => stat.is_plottable);
      controls.innerHTML = '';
      plottedStats.forEach(stat => {{
        const label = document.createElement('label');
        label.className = 'metric-option';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = stat.angle;
        input.checked = defaults.has(stat.angle);
        input.addEventListener('change', () => {{
          renderMetricCards(currentIndex);
          applyMetricSelection(currentIndex);
        }});
        label.appendChild(input);
        label.appendChild(document.createTextNode(stat.angle));
        controls.appendChild(label);
      }});
      applyMetricSelection(index);
    }}

    function applyMetricSelection(index) {{
      const selected = new Set(selectedNames());
      const traces = figPayload[index].data || [];
      const visible = traces.map(trace => selected.has(trace.name) ? true : 'legendonly');
      if (visible.length) Plotly.restyle(plot, 'visible', visible);
    }}

    function renderMetricCards(index) {{
      const selected = new Set(selectedNames());
      const rows = motionPayload[index].stats.filter(stat => stat.is_plottable && selected.has(stat.angle));
      if (!rows.length) {{
        cards.innerHTML = '<p class="muted">未选择重点指标。勾选上方指标后，将在此处显示 ROM、峰值时间和统计摘要。</p>';
        return;
      }}
      cards.innerHTML = rows.map(stat => `
        <article class="metric-card">
          <div class="label">${{escapeHtml(stat.kind_short)}} · ${{escapeHtml(stat.movement_label)}}</div>
          <div class="value">${{fmt(stat.rom)}}° ROM</div>
          <h3><span>${{escapeHtml(stat.angle)}}</span> <button class="icon-button metric-info" type="button" data-angle="${{escapeHtml(stat.angle)}}" aria-label="查看 ${{escapeHtml(stat.angle)}} 的指标解释">i</button></h3>
          <p class="muted">${{escapeHtml(stat.description)}}</p>
          <dl>
            <dt>最小值</dt><dd>${{fmt(stat.min)}}° <button class="seek-button" data-time="${{stat.time_at_min}}">到 ${{fmt(stat.time_at_min, 2)}}s</button></dd>
            <dt>最大值</dt><dd>${{fmt(stat.max)}}° <button class="seek-button" data-time="${{stat.time_at_max}}">到 ${{fmt(stat.time_at_max, 2)}}s</button></dd>
            <dt>均值</dt><dd>${{fmt(stat.mean)}}°</dd>
            <dt>标准差</dt><dd>${{fmt(stat.std)}}°</dd>
          </dl>
        </article>
      `).join('');
    }}

    function findStat(angle) {{
      return [...(motionPayload[currentIndex].stats || []), ...(motionPayload[currentIndex].auxiliary_stats || [])]
        .find(stat => stat.angle === angle);
    }}

    function openModal(modal) {{
      if (!modal) return;
      modal.hidden = false;
      const close = modal.querySelector('[data-close-modal]');
      if (close) close.focus();
    }}

    function closeModal(modal) {{
      if (modal) modal.hidden = true;
    }}

    function openMetricModal(stat) {{
      if (!stat) return;
      metricModalTitle.textContent = stat.angle;
      metricModalContent.innerHTML = `
        <dl>
          <dt>动作含义</dt><dd>${{escapeHtml(stat.movement_label)}}</dd>
          <dt>数据来源</dt><dd>${{escapeHtml(stat.source || stat.kind)}}。${{escapeHtml(stat.kind_note || '')}}</dd>
          <dt>指标类型</dt><dd>${{stat.is_auxiliary ? '辅助数据，不作为核心活动度指标' : (stat.is_angle ? '角度指标' : '非角度指标')}}；单位：${{escapeHtml(stat.unit || 'deg')}}</dd>
          <dt>原始来源</dt><dd>${{escapeHtml(stat.technical_source || stat.raw_metric || '')}}</dd>
          <dt>运动平面</dt><dd>${{escapeHtml(stat.movement_plane || '')}}</dd>
          <dt>0°位/中立位</dt><dd>${{escapeHtml(stat.neutral_definition || '未提供中立位定义。')}}</dd>
          <dt>数值方向</dt><dd>${{escapeHtml(stat.direction_definition || '未提供数值方向定义。')}}</dd>
          <dt>计算定义</dt><dd>${{escapeHtml(stat.description)}}</dd>
          <dt>解释边界</dt><dd>${{escapeHtml(stat.interpretation)}}</dd>
          <dt>拍摄平面提示</dt><dd>${{escapeHtml(stat.camera_note)}}</dd>
          <dt>${{escapeHtml(stat.range_label || 'ROM')}} 说明</dt><dd>${{escapeHtml(stat.rom_note)}}</dd>
        </dl>
      `;
      openModal(metricModal);
    }}

    motionPayload.forEach((item, index) => {{
      const button = document.createElement('button');
      button.textContent = item.name + ' · ' + item.kind_short;
      button.title = item.kind_note;
      button.onclick = () => renderMotion(index);
      tabs.appendChild(button);
    }});
    cards.addEventListener('click', event => {{
      const button = event.target.closest('button[data-time]');
      if (button) {{
        seekTo(button.dataset.time);
        return;
      }}
      const info = event.target.closest('button.metric-info');
      if (info) openMetricModal(findStat(info.dataset.angle));
    }});
    statsTable.addEventListener('click', event => {{
      const info = event.target.closest('button.metric-info');
      if (info) openMetricModal(findStat(info.dataset.angle));
    }});
    auxiliaryTable.addEventListener('click', event => {{
      const info = event.target.closest('button.metric-info');
      if (info) openMetricModal(findStat(info.dataset.angle));
    }});
    document.getElementById('openQualityDetail')?.addEventListener('click', () => openModal(qualityModal));
    document.querySelectorAll('[data-close-modal]').forEach(button => {{
      button.addEventListener('click', () => closeModal(button.closest('.modal-backdrop')));
    }});
    document.querySelectorAll('.modal-backdrop').forEach(modal => {{
      modal.addEventListener('click', event => {{
        if (event.target === modal) closeModal(modal);
      }});
    }});
    document.addEventListener('keydown', event => {{
      if (event.key === 'Escape') {{
        closeModal(metricModal);
        closeModal(qualityModal);
      }}
    }});
    if (figPayload.length) renderMotion(0);
  </script>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")


def write_excel_report(
    excel_path: Path,
    motions: list[tuple[Path, pd.DataFrame]],
    config: dict[str, Any],
    environment_report: dict[str, Any],
    output_dir: Path,
    quality: dict[str, Any] | None = None,
) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    quality = quality or collect_quality_diagnostics(output_dir, config)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_rows = []
        for path, df in motions:
            stats = angle_statistics(df, classify_motion_file(path))
            stats.insert(0, "source", str(path.relative_to(output_dir)))
            summary_rows.append(stats)
            df.to_excel(writer, sheet_name=_sheet_name(path.stem), index=False)
        if summary_rows:
            pd.concat(summary_rows, ignore_index=True).to_excel(writer, sheet_name="角度统计", index=False)
        _quality_dataframe(quality).to_excel(writer, sheet_name="质量诊断", index=False)
        _video_metadata_dataframe(quality.get("video_metadata", [])).to_excel(
            writer, sheet_name="视频元数据", index=False
        )
        pd.DataFrame(
            [{"key": k, "value": str(v)} for k, v in flatten_dict(config).items()]
        ).to_excel(writer, sheet_name="参数", index=False)
        pd.DataFrame(
            [
                {
                    "key": k,
                    "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v,
                }
                for k, v in environment_report.items()
            ]
        ).to_excel(writer, sheet_name="环境", index=False)
        files = [
            {"path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size}
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        ]
        pd.DataFrame(files).to_excel(writer, sheet_name="输出文件", index=False)


def angle_statistics(
    df: pd.DataFrame,
    kind: dict[str, str] | None = None,
    quality: dict[str, Any] | None = None,
    report_details: bool = False,
    metric_metadata: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = []
    if "time" not in df.columns:
        return pd.DataFrame(rows)
    kind = kind or {"kind_short": "数据", "kind": "运动数据", "kind_note": ""}
    camera_note = _camera_plane_note(quality)
    for column in df.columns:
        if column == "time":
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if series.dropna().empty:
            continue
        min_idx = series.idxmin()
        max_idx = series.idxmax()
        explicit_meta = (metric_metadata or {}).get(column)
        meta = explicit_meta or measure_metadata(column, kind, camera_note)
        display_metric = str(meta.get("display_metric") or column) if explicit_meta else column
        row = {
            "angle": display_metric,
            "display_name": meta["display_name"],
            "kind_short": kind["kind_short"],
            "movement_label": meta["movement_label"],
            "description": meta["description"],
            "interpretation": meta["interpretation"],
            "_sort_key": (
                meta["body_region_rank"],
                meta["side_rank"],
                meta["motion_rank"],
                column.lower(),
            ),
            "min": float(series.min()),
            "time_at_min": float(df.loc[min_idx, "time"]),
            "max": float(series.max()),
            "time_at_max": float(df.loc[max_idx, "time"]),
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series.dropna()) > 1 else 0.0,
            "rom": float(series.max() - series.min()),
        }
        if report_details:
            row.update(
                {
                    "kind": kind.get("kind", kind["kind_short"]),
                    "kind_note": kind.get("kind_note", ""),
                    "source": meta["source"],
                    "unit": meta["unit"],
                    "is_angle": meta["is_angle"],
                    "is_auxiliary": meta["is_auxiliary"],
                    "is_plottable": meta["is_plottable"],
                    "body_region_rank": meta["body_region_rank"],
                    "side_rank": meta["side_rank"],
                    "motion_rank": meta["motion_rank"],
                    "action_label": meta["action_label"],
                    "explanation": meta["explanation"],
                    "raw_metric": meta.get("raw_metric", column),
                    "display_metric": display_metric,
                    "movement_plane": meta.get("movement_plane", ""),
                    "neutral_definition": meta.get("neutral_definition", ""),
                    "direction_definition": meta.get("direction_definition", ""),
                    "technical_source": meta.get("technical_source", meta["source"]),
                    "is_primary_rom_metric": bool(meta.get("is_primary_rom_metric", meta["is_angle"] and not meta["is_auxiliary"])),
                    "range_label": "ROM"
                    if bool(meta.get("is_primary_rom_metric", meta["is_angle"] and not meta["is_auxiliary"]))
                    else "Range",
                    "camera_note": meta["camera_note"],
                    "rom_note": STANDARD_ROM_NOTE
                    if bool(meta.get("is_primary_rom_metric", meta["is_angle"] and not meta["is_auxiliary"]))
                    else AUXILIARY_RANGE_NOTE,
                }
            )
        rows.append(row)
    if report_details:
        rows.sort(key=lambda item: item["_sort_key"])
    for row in rows:
        row.pop("_sort_key", None)
    return pd.DataFrame(rows)


def classify_motion_file(path: Path) -> dict[str, str]:
    stem = path.stem.lower()
    if stem.endswith("_ik") or "_ik" in stem:
        return {
            "kind_short": "OpenSim IK 关节活动角（高级）",
            "kind": "OpenSim IK 关节活动角（高级）",
            "kind_note": "这是 OpenSim 逆运动学输出。旋转类 coordinate 按关节角时间序列解释；平移和辅助约束列仅作为高级诊断数据，不代表关节活动度。",
        }
    return {
        "kind_short": "Sports2D 2D 平面活动角",
        "kind": "Sports2D 2D 平面活动角",
        "kind_note": "这是由视频平面关键点计算或整理的 2D 活动角，不是完整三维解剖角。",
    }


def measure_metadata(
    name: str,
    kind: dict[str, str],
    camera_note: str | None = None,
) -> dict[str, Any]:
    display = name.strip()
    lower = name.lower()
    camera_note = camera_note or kind.get("camera_note") or _camera_plane_note(None)
    if str(kind["kind_short"]).startswith("OpenSim IK"):
        return _opensim_measure_metadata(name, display, lower, camera_note)
    return _sports2d_measure_metadata(name, display, lower, camera_note)


def _side_metadata(lower: str) -> tuple[str, int]:
    if lower.startswith("left ") or lower.endswith("_l") or "_l_" in lower:
        return "左侧", 0
    if lower.startswith("right ") or lower.endswith("_r") or "_r_" in lower:
        return "右侧", 1
    return "中线/整体", 2


def _metric_payload(
    display: str,
    *,
    source: str,
    unit: str,
    is_angle: bool,
    is_auxiliary: bool,
    body_region_rank: int,
    side_rank: int,
    motion_rank: int,
    action_label: str,
    description: str,
    interpretation: str,
    camera_note: str,
    raw_metric: str | None = None,
    display_metric: str | None = None,
    movement_plane: str = "",
    neutral_definition: str = "",
    direction_definition: str = "",
    technical_source: str = "",
    is_primary_rom_metric: bool | None = None,
) -> dict[str, Any]:
    primary = bool(is_angle and not is_auxiliary) if is_primary_rom_metric is None else is_primary_rom_metric
    return {
        "display_name": display,
        "raw_metric": raw_metric or display,
        "display_metric": display_metric or display,
        "movement_label": action_label,
        "action_label": action_label,
        "description": description,
        "explanation": description,
        "interpretation": f"{interpretation} {camera_note}".strip(),
        "source": source,
        "unit": unit,
        "is_angle": is_angle,
        "is_auxiliary": is_auxiliary,
        "is_plottable": bool(is_angle and not is_auxiliary),
        "body_region_rank": body_region_rank,
        "side_rank": side_rank,
        "motion_rank": motion_rank,
        "camera_note": camera_note,
        "movement_plane": movement_plane,
        "neutral_definition": neutral_definition,
        "direction_definition": direction_definition,
        "technical_source": technical_source or source,
        "is_primary_rom_metric": bool(primary),
    }


def _sports2d_measure_metadata(name: str, display: str, lower: str, camera_note: str) -> dict[str, str]:
    side, side_rank = _side_metadata(lower)
    source = "Sports2D 2D 平面角"
    if "ankle" in lower:
        rank = BODY_RANKS["ankle"]
        label = f"{side}踝关节背屈/跖屈平面角"
        desc = "Sports2D 使用足跟、大脚趾、踝和膝等二维关键点在视频画面中的几何关系计算踝关节平面角。该值描述的是画面平面内的小腿与足部之间的夹角变化。"
        interp = "在标准侧向拍摄且运动主要发生在矢状面时，可作为踝背屈/跖屈趋势的参考；在正面、背面或明显斜向拍摄时，不应解释为真实三维踝关节背屈角或跖屈角。"
    elif "knee" in lower:
        rank = BODY_RANKS["knee"]
        label = f"{side}膝关节屈曲/伸展平面角"
        desc = "Sports2D 根据髋、膝、踝三个二维关键点计算膝关节在视频平面内的夹角。该指标反映大腿与小腿在画面中的相对折叠程度。"
        interp = "侧向拍摄时通常用于描述膝屈曲/伸展趋势；正面或背面拍摄时，它更接近画面内的投影夹角，不能直接等同于矢状面膝屈曲角。"
    elif "hip" in lower:
        rank = BODY_RANKS["hip"]
        label = f"{side}髋关节屈曲/伸展平面角"
        desc = "Sports2D 以膝、髋、肩三个二维关键点形成的夹角描述髋部相对躯干和下肢的画面内角度变化。"
        interp = "侧向拍摄且躯干与下肢主要在矢状面运动时，可用于观察髋屈曲/伸展趋势；正面、背面或斜向拍摄会混入髋外展、躯干侧倾和透视投影影响。"
    elif "shoulder" in lower:
        rank = BODY_RANKS["shoulder"]
        label = f"{side}肩关节屈曲/伸展平面角"
        desc = "Sports2D 通过髋、肩、肘三个二维关键点计算上臂相对躯干的画面内夹角。"
        interp = "该指标适合观察上臂相对躯干在视频平面内的运动趋势；它不是肩关节三维屈曲、外展、内外旋的分解结果。"
    elif "elbow" in lower:
        rank = BODY_RANKS["elbow"]
        label = f"{side}肘关节屈曲/伸展平面角"
        desc = "Sports2D 根据腕、肘、肩三个二维关键点计算肘关节在视频平面内的夹角。"
        interp = "多数情况下可用于描述肘屈曲/伸展趋势，但前臂旋前/旋后、遮挡和透视投影会改变二维关键点位置，从而影响该角度。"
    elif "wrist" in lower:
        rank = BODY_RANKS["wrist"]
        label = f"{side}腕关节平面角"
        desc = "该指标来自腕部及相邻肢段关键点在视频平面中的几何关系，通常比髋、膝、踝等大关节更容易受到关键点检测误差影响。"
        interp = "腕部角度应结合处理后视频逐段核对，尤其是在手部遮挡、快速摆动或器械遮挡明显时，不宜单独作为结论依据。"
    elif any(segment in lower for segment in ["foot", "shank", "thigh", "pelvis", "trunk", "head", "arm", "forearm"]):
        rank = _body_rank_from_lower(lower)
        segment_name = _segment_label(lower)
        label = f"{side}{segment_name}相对水平线角度"
        desc = "节段角表示对应身体节段在视频平面中相对水平线的方向角，通常按画面坐标系中的逆时针方向计算。"
        interp = "节段角描述节段姿态，不是关节夹角。若摄像机存在倾斜、视频被旋转或地面角校正不准确，节段角会首先受到影响。"
    else:
        rank = 110
        label = "视频平面角"
        desc = "该列来自 Sports2D 输出的运动学文件，表示由二维关键点或节段方向计算得到的视频平面几何角。"
        interp = "应结合列名、拍摄方向、处理后视频和质量诊断解释该指标，不应自动视为三维解剖角或可靠的 OpenSim IK 结果。"
    return _metric_payload(
        display,
        source=source,
        unit="deg",
        is_angle=True,
        is_auxiliary=False,
        body_region_rank=rank,
        side_rank=side_rank,
        motion_rank=0,
        action_label=label,
        description=desc,
        interpretation=interp,
        camera_note=camera_note,
        raw_metric=name,
        display_metric=label,
        movement_plane="视频平面",
        neutral_definition="未标准化：该指标来自 Sports2D 原始 .mot 平面角，未能从关键点几何重新定义 0°中立位。",
        direction_definition="数值方向沿用 Sports2D 原始输出；除非已在报告中标注为标准化 2D 活动角，否则不应直接解释为关节活动度。",
        technical_source=f"Sports2D 原始 MOT 列：{name}",
    )


def _opensim_measure_metadata(name: str, display: str, lower: str, camera_note: str) -> dict[str, str]:
    side, side_rank = _side_metadata(lower)
    source = "OpenSim IK"
    is_angle = True
    is_auxiliary = False
    unit = "deg"
    rank = 110
    motion_rank = 8
    label = "OpenSim IK 角度"
    desc = "该列来自 OpenSim 逆运动学 MOT 文件。旋转类 coordinate 在本报告中按关节角时间序列解释。"
    interp = "只有当 marker error、人体尺度、拍摄方向、标定和模型匹配均合理时，才建议进一步按三维关节角解释。若质量诊断提示异常，应优先查看处理后视频中的 2D 骨架，并以 Sports2D 2D 平面活动角作为主报告结果。"

    if lower in {"pelvis_tx", "pelvis_ty", "pelvis_tz"}:
        axis = {"pelvis_tx": "前后", "pelvis_ty": "上下", "pelvis_tz": "左右"}[lower]
        is_angle = False
        is_auxiliary = True
        unit = "m"
        rank = BODY_RANKS["pelvis"]
        motion_rank = 3
        label = f"骨盆{axis}平移（辅助数据，非关节活动度）"
        desc = "该列是 OpenSim 骨盆平移坐标，单位通常为米，用于描述模型整体位置变化。它不是关节角，也不是关节活动度。"
    elif lower in {"pelvis_tilt", "pelvis_list", "pelvis_rotation"}:
        rank = BODY_RANKS["pelvis"]
        mapping = {
            "pelvis_tilt": ("骨盆前倾/后倾角", 0),
            "pelvis_list": ("骨盆左右倾斜角", 1),
            "pelvis_rotation": ("骨盆轴向旋转角", 2),
        }
        label, motion_rank = mapping[lower]
        desc = f"该列是 OpenSim IK 估计的{label}时间序列，反映骨盆相对 OpenSim 参考坐标系的旋转姿态。"
    elif "hip_flexion" in lower:
        rank = BODY_RANKS["hip"]
        motion_rank = 0
        label = f"{side}髋关节屈曲/伸展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}髋关节屈曲/伸展角时间序列。ROM 表示当前片段内该角度的最大值减最小值。"
    elif "hip_adduction" in lower:
        rank = BODY_RANKS["hip"]
        motion_rank = 1
        label = f"{side}髋关节内收/外展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}髋关节额状面内收/外展角时间序列。"
    elif "hip_rotation" in lower:
        rank = BODY_RANKS["hip"]
        motion_rank = 2
        label = f"{side}髋关节内旋/外旋角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}髋关节轴向旋转角时间序列。"
    elif re.search(r"knee_angle_[rl]$", lower):
        rank = BODY_RANKS["knee"]
        motion_rank = 0
        label = f"{side}膝关节屈曲/伸展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}膝关节屈曲/伸展角时间序列。ROM 表示当前片段内膝屈伸角的观测变化范围。"
    elif "knee_angle" in lower and "beta" in lower:
        rank = BODY_RANKS["knee"]
        motion_rank = 9
        is_auxiliary = True
        label = f"{side}膝关节辅助约束角（非核心活动度）"
        desc = "该列是 OpenSim 膝关节模型的辅助约束坐标，不作为用户报告中的核心膝关节活动度指标。"
    elif "ankle_angle" in lower:
        rank = BODY_RANKS["ankle"]
        motion_rank = 0
        label = f"{side}踝关节背屈/跖屈角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}踝关节背屈/跖屈角时间序列。"
    elif "subtalar_angle" in lower:
        rank = BODY_RANKS["subtalar"]
        motion_rank = 1
        label = f"{side}距下关节内翻/外翻角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}距下关节内翻/外翻角时间序列。"
    elif "mtp_angle" in lower:
        rank = BODY_RANKS["mtp"]
        motion_rank = 0
        label = f"{side}跖趾关节屈曲/伸展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}跖趾关节屈曲/伸展角时间序列。"
    elif any(token in lower for token in ["flex_ext", "lat_bending", "axial_rotation"]):
        rank = BODY_RANKS["spine"]
        segment = name.rsplit("_", 2)[0].replace("_", "-")
        if "flex_ext" in lower:
            label = f"{segment} 脊柱屈曲/伸展角（OpenSim IK）"
            motion_rank = 0
        elif "lat_bending" in lower:
            label = f"{segment} 脊柱侧屈角（OpenSim IK）"
            motion_rank = 1
        else:
            label = f"{segment} 脊柱轴向旋转角（OpenSim IK）"
            motion_rank = 2
        desc = f"该列是 OpenSim IK 估计的{label.replace('（OpenSim IK）', '')}时间序列。"
    elif lower.startswith("abs_t"):
        is_angle = False
        is_auxiliary = True
        unit = "m"
        rank = BODY_RANKS["trunk"]
        motion_rank = 3
        label = "躯干/腹部模型辅助平移（非关节活动度）"
        desc = "该列是 OpenSim 模型的辅助平移坐标，不是关节角，也不应作为关节 ROM 解读。"
    elif lower.startswith("abs_r"):
        is_auxiliary = True
        rank = BODY_RANKS["trunk"]
        motion_rank = 9
        label = "躯干/腹部模型辅助旋转角（非核心活动度）"
        desc = "该列是 OpenSim 模型的辅助旋转坐标，通常用于模型求解，不作为核心关节活动度指标展示。"
    elif lower == "neck_flexion":
        rank = BODY_RANKS["neck"]
        motion_rank = 0
        label = "颈部屈曲/伸展角（OpenSim IK）"
        desc = "该列是 OpenSim IK 估计的颈部屈曲/伸展角时间序列。"
    elif lower == "neck_bending":
        rank = BODY_RANKS["neck"]
        motion_rank = 1
        label = "颈部侧屈角（OpenSim IK）"
        desc = "该列是 OpenSim IK 估计的颈部侧屈角时间序列。"
    elif lower == "neck_rotation":
        rank = BODY_RANKS["neck"]
        motion_rank = 2
        label = "颈部旋转角（OpenSim IK）"
        desc = "该列是 OpenSim IK 估计的颈部轴向旋转角时间序列。"
    elif "arm_flex" in lower:
        rank = BODY_RANKS["shoulder"]
        motion_rank = 0
        label = f"{side}肩关节屈曲/伸展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}肩关节屈曲/伸展角时间序列。"
    elif "arm_add" in lower:
        rank = BODY_RANKS["shoulder"]
        motion_rank = 1
        label = f"{side}肩关节内收/外展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}肩关节内收/外展角时间序列。"
    elif "arm_rot" in lower:
        rank = BODY_RANKS["shoulder"]
        motion_rank = 2
        label = f"{side}肩关节内旋/外旋角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}肩关节旋转角时间序列。"
    elif "elbow_flex" in lower:
        rank = BODY_RANKS["elbow"]
        motion_rank = 0
        label = f"{side}肘关节屈曲/伸展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}肘关节屈曲/伸展角时间序列。"
    elif "pro_sup" in lower:
        rank = BODY_RANKS["forearm"]
        motion_rank = 2
        label = f"{side}前臂旋前/旋后角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}前臂旋前/旋后角时间序列。"
    elif "wrist_flex" in lower:
        rank = BODY_RANKS["wrist"]
        motion_rank = 0
        label = f"{side}腕关节屈曲/伸展角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}腕关节屈曲/伸展角时间序列。"
    elif "wrist_dev" in lower:
        rank = BODY_RANKS["wrist"]
        motion_rank = 1
        label = f"{side}腕关节桡偏/尺偏角（OpenSim IK）"
        desc = f"该列是 OpenSim IK 估计的{side}腕关节桡偏/尺偏角时间序列。"
    elif lower.endswith(("_tx", "_ty", "_tz")):
        is_angle = False
        is_auxiliary = True
        unit = "m"
        motion_rank = 3
        label = "OpenSim 平移辅助坐标（非关节活动度）"
        desc = "该列是 OpenSim 平移坐标，不是关节角，也不应作为关节 ROM 解读。"

    movement_plane = _opensim_movement_plane(lower)
    neutral_definition, direction_definition = _opensim_neutral_direction(
        lower, is_angle=is_angle, is_auxiliary=is_auxiliary
    )
    return _metric_payload(
        display,
        source=source,
        unit=unit,
        is_angle=is_angle,
        is_auxiliary=is_auxiliary,
        body_region_rank=rank,
        side_rank=side_rank,
        motion_rank=motion_rank,
        action_label=label,
        description=desc,
        interpretation=interp,
        camera_note=camera_note,
        raw_metric=name,
        display_metric=label,
        movement_plane=movement_plane,
        neutral_definition=neutral_definition,
        direction_definition=direction_definition,
        technical_source=f"OpenSim IK MOT coordinate：{name}",
        is_primary_rom_metric=bool(is_angle and not is_auxiliary),
    )


def _opensim_movement_plane(lower: str) -> str:
    if any(token in lower for token in ["flexion", "knee_angle", "ankle_angle", "mtp_angle", "flex_ext", "arm_flex", "elbow_flex", "wrist_flex", "pelvis_tilt", "neck_flexion"]):
        return "OpenSim 矢状面/屈伸方向"
    if any(token in lower for token in ["adduction", "list", "lat_bending", "arm_add", "wrist_dev", "subtalar"]):
        return "OpenSim 额状面/内外侧方向"
    if any(token in lower for token in ["rotation", "axial_rotation", "arm_rot", "pro_sup"]):
        return "OpenSim 水平面/轴向旋转方向"
    if lower.endswith(("_tx", "_ty", "_tz")) or lower.startswith("abs_t"):
        return "OpenSim 平移辅助坐标"
    return "OpenSim 未分类旋转方向"


def _opensim_neutral_direction(
    lower: str,
    *,
    is_angle: bool,
    is_auxiliary: bool,
) -> tuple[str, str]:
    if not is_angle or lower.endswith(("_tx", "_ty", "_tz")) or lower.startswith("abs_t"):
        return (
            "不适用。该指标为平移或辅助量，不定义关节 0°中立位。",
            "数值增大表示 OpenSim 原始坐标对应方向的平移增加，不代表关节活动角增大。",
        )
    if is_auxiliary:
        return (
            "不作为主报告关节中立位解释。该指标属于 OpenSim 模型求解或约束相关辅助坐标。",
            "数值方向沿用 OpenSim 原始坐标定义，不建议作为临床或训练反馈指标。",
        )
    if "knee_angle" in lower:
        return (
            "0°通常接近 OpenSim 模型中的膝关节伸直位。",
            "数值增大通常表示膝关节屈曲增加；数值减小表示更接近伸直或相对伸展。",
        )
    if "hip_flexion" in lower:
        return (
            "0°对应 OpenSim 模型髋关节屈伸坐标的解剖中立位。",
            "数值增大通常表示髋屈曲增加；数值减小或负值通常表示相对伸展。",
        )
    if "hip_adduction" in lower:
        return (
            "0°对应 OpenSim 模型髋关节额状面中立位。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述髋内收/外展方向的偏移。",
        )
    if "hip_rotation" in lower:
        return (
            "0°对应 OpenSim 模型髋关节轴向旋转中立位。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述髋内旋/外旋方向的偏移。",
        )
    if "ankle_angle" in lower:
        return (
            "0°通常接近 OpenSim 模型踝关节中立位。",
            "数值增大通常表示背屈增加；数值减小或负值通常表示跖屈增加。",
        )
    if "subtalar_angle" in lower:
        return (
            "0°对应 OpenSim 模型距下关节中立位。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述足部内翻/外翻方向的偏移。",
        )
    if "mtp_angle" in lower:
        return (
            "0°对应 OpenSim 模型跖趾关节中立位。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述跖趾屈曲/伸展方向的偏移。",
        )
    if "pelvis_tilt" in lower:
        return (
            "0°对应 OpenSim 模型骨盆前后倾坐标的参考中立姿态。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述骨盆前倾/后倾姿态变化。",
        )
    if "pelvis_list" in lower:
        return (
            "0°对应 OpenSim 模型骨盆左右倾斜坐标的参考中立姿态。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述骨盆向左或向右倾斜。",
        )
    if "pelvis_rotation" in lower:
        return (
            "0°对应 OpenSim 模型骨盆轴向旋转坐标的参考中立姿态。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述骨盆向左或向右旋转。",
        )
    if any(token in lower for token in ["flex_ext", "neck_flexion", "arm_flex", "elbow_flex", "wrist_flex"]):
        return (
            "0°对应 OpenSim 模型该关节或节段屈伸坐标的中立位。",
            "数值增大通常表示屈曲方向增加；数值减小通常表示伸展方向增加或更接近中立位。",
        )
    if any(token in lower for token in ["lat_bending", "neck_bending", "arm_add", "wrist_dev"]):
        return (
            "0°对应 OpenSim 模型该关节或节段内外侧方向的中立位。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述侧屈、内收/外展或桡偏/尺偏方向的偏移。",
        )
    if any(token in lower for token in ["axial_rotation", "neck_rotation", "arm_rot", "pro_sup"]):
        return (
            "0°对应 OpenSim 模型该关节或节段轴向旋转坐标的中立位。",
            "数值正负方向沿用 OpenSim 原始坐标定义，用于描述内外旋、旋前/旋后或轴向旋转方向的偏移。",
        )
    return (
        "0°按 OpenSim 模型该旋转坐标的中立位定义。",
        "数值方向沿用 OpenSim 原始坐标定义；解释时应结合模型文档、质量诊断和处理后视频复核。",
    )


def _segment_label(lower: str) -> str:
    labels = {
        "foot": "足部节段",
        "shank": "小腿节段",
        "thigh": "大腿节段",
        "pelvis": "骨盆节段",
        "trunk": "躯干节段",
        "shoulders": "肩带节段",
        "head": "头部节段",
        "forearm": "前臂节段",
        "arm": "上臂节段",
    }
    for key, value in labels.items():
        if key in lower:
            return value
    return "身体节段"


def _body_rank_from_lower(lower: str) -> int:
    for key in sorted(BODY_RANKS, key=len, reverse=True):
        if key in lower:
            return BODY_RANKS[key]
    return 110


def _camera_plane_note(quality: dict[str, Any] | None) -> str:
    if not quality:
        return (
            "当前报告未获得明确的拍摄方向诊断；请以处理后视频中的骨架方向、受试者可见侧和实际拍摄机位为准。"
            "在拍摄方向无法确认时，应将 Sports2D 原生角度解释为视频平面角，而不是特定解剖平面的屈伸、外展或旋转角。"
        )
    insights = quality.get("run_log_insights", {}) or {}
    seen_from = str(insights.get("sports2d_seen_from", "")).strip().lower()
    configured = str(insights.get("configured_visible_side", "")).strip().lower()
    seen_tokens = seen_from.split()
    configured_tokens = configured.replace(",", " ").split()
    if any(token in seen_tokens for token in ["front", "back"]) or any(
        token in configured_tokens for token in ["front", "back"]
    ):
        return (
            "质量诊断提示当前数据接近正面或背面视角。此时髋、膝、踝等指标主要反映画面平面内的投影几何关系，"
            "不应解释为矢状面屈曲/伸展角；它们更适合用于观察左右对称性、额状面趋势或关键点稳定性。"
        )
    if any(token in seen_tokens for token in ["right", "left", "side"]) or any(
        token in configured_tokens for token in ["right", "left", "side"]
    ):
        return (
            "质量诊断提示当前数据接近侧向视角。若摄像机基本垂直于受试者运动平面，髋、膝、踝等下肢指标通常可作为"
            "矢状面屈曲/伸展趋势的二维近似；若存在明显斜拍、转体或出平面运动，仍应按视频平面投影角解释。"
        )
    if "auto" in configured_tokens or "auto" in seen_tokens:
        return (
            "可见侧设置包含 auto，拍摄平面需要通过处理后视频和日志人工复核。若 Sports2D 的方向判断与真实机位不一致，"
            "指标解释应以实际机位为准，并避免直接套用矢状面或额状面术语。"
        )
    return (
        "当前报告未获得明确的拍摄方向诊断；请结合处理后视频、受试者可见侧和实际摄像机位置解释。"
        "在方向不确定时，Sports2D 原生指标应统一解释为视频平面内的二维角度。"
    )


def _load_motion_files(sports_dir: Path) -> list[tuple[Path, pd.DataFrame]]:
    motions = []
    for mot_path in sorted(sports_dir.rglob("*.mot")):
        try:
            df = read_mot(mot_path)
        except Exception:
            continue
        if "time" in df.columns:
            motions.append((mot_path, df))
    return motions


def _standardize_report_motion(
    path: Path,
    df: pd.DataFrame,
    kind: dict[str, Any],
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    if str(kind["kind_short"]).startswith("OpenSim IK"):
        return _standardize_ik_motion(path, df, kind, quality)
    trc_path = _find_px_trc_for_motion(path)
    if trc_path is not None:
        trc_df = read_trc(trc_path)
        standardized = _standardize_2d_from_trc(trc_path, trc_df, quality)
        if standardized is not None:
            return standardized
    return _standardize_raw_2d_fallback(path, df, kind, quality)


def _standardize_ik_motion(
    path: Path,
    df: pd.DataFrame,
    kind: dict[str, Any],
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    camera_note = _camera_plane_note(quality)
    source_kind = {
        **kind,
        "kind_short": "OpenSim IK 关节活动角（高级）",
        "kind": "OpenSim IK 关节活动角（高级）",
        "kind_note": (
            "OpenSim IK 旋转类 coordinate 已按关节活动角展示；平移和辅助约束坐标不进入主曲线，"
            "仅保留在高级诊断附录中。IK 结果的可信度必须结合 marker error、拍摄方向、尺度和模型匹配情况判断。"
        ),
        "camera_note": camera_note,
    }
    display = pd.DataFrame({"time": pd.to_numeric(df["time"], errors="coerce")})
    metric_metadata: dict[str, dict[str, Any]] = {}
    auxiliary = pd.DataFrame({"time": display["time"]})
    auxiliary_metadata: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        if column == "time":
            continue
        meta = measure_metadata(column, source_kind, camera_note)
        values = pd.to_numeric(df[column], errors="coerce")
        if meta.get("is_primary_rom_metric") and meta.get("is_plottable"):
            metric_name = _unique_column_name(display, str(meta["display_metric"]))
            display[metric_name] = values
            metric_metadata[metric_name] = {**meta, "display_metric": metric_name, "display_name": metric_name}
        else:
            auxiliary[column] = values
            auxiliary_metadata[column] = {
                **meta,
                "display_metric": column,
                "display_name": column,
                "is_plottable": False,
            }

    auxiliary_stats = angle_statistics(
        auxiliary,
        source_kind,
        quality,
        report_details=True,
        metric_metadata=auxiliary_metadata,
    )
    return {
        "df": display,
        "kind": source_kind,
        "metric_metadata": metric_metadata,
        "auxiliary_stats": auxiliary_stats,
        "source_note": f"OpenSim IK MOT：{path.name}",
    }


def _standardize_raw_2d_fallback(
    path: Path,
    df: pd.DataFrame,
    kind: dict[str, Any],
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    camera_note = _camera_plane_note(quality)
    fallback_kind = {
        **kind,
        "kind_short": "Sports2D 2D 原始平面角（未标准化）",
        "kind": "Sports2D 2D 原始平面角（未标准化）",
        "kind_note": (
            "未找到可用于重新计算活动角的像素 TRC 文件，因此本标签保留 Sports2D 原始 .mot 平面角。"
            "这些值未统一为 0°中立位，不建议直接解释为关节活动度。"
        ),
        "camera_note": camera_note,
    }
    display = df.copy()
    for column in display.columns:
        display[column] = pd.to_numeric(display[column], errors="coerce")
    metadata: dict[str, dict[str, Any]] = {}
    for column in display.columns:
        if column == "time":
            continue
        meta = measure_metadata(column, fallback_kind, camera_note)
        metadata[column] = {
            **meta,
            "raw_metric": column,
            "display_metric": column,
            "source": "Sports2D 原始 2D 平面角（未标准化）",
            "technical_source": f"Sports2D 原始 MOT 列：{column}",
            "neutral_definition": "未标准化：缺少对应 px TRC，无法可靠给出 0°中立位。",
            "direction_definition": "数值方向沿用 Sports2D 原始输出；请勿直接解读为关节活动度。",
            "is_primary_rom_metric": True,
        }
    auxiliary_stats = pd.DataFrame()
    return {
        "df": display,
        "kind": fallback_kind,
        "metric_metadata": metadata,
        "auxiliary_stats": auxiliary_stats,
        "source_note": f"Sports2D 原始 MOT：{path.name}",
    }


def _standardize_2d_from_trc(
    trc_path: Path,
    trc_df: pd.DataFrame,
    quality: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if trc_df.empty or "time" not in trc_df.columns:
        return None
    camera_note = _camera_plane_note(quality)
    kind = {
        "kind_short": "Sports2D 2D 平面活动角",
        "kind": "Sports2D 2D 平面活动角",
        "kind_note": (
            "这些指标由 Sports2D 像素 TRC 关键点重新计算，并统一为视频平面内的关节活动角。"
            "2D 中立位服务于视频平面解释，不等同于完整三维解剖测量。"
        ),
        "camera_note": camera_note,
    }
    display = pd.DataFrame({"time": pd.to_numeric(trc_df["time"], errors="coerce")})
    metadata: dict[str, dict[str, Any]] = {}

    def add_metric(
        column_name: str,
        series: pd.Series,
        *,
        side: str,
        body_key: str,
        motion_rank: int,
        raw_metric: str,
        action_label: str,
        description: str,
        neutral_definition: str,
        direction_definition: str,
        movement_plane: str = "视频平面",
    ) -> None:
        values = pd.to_numeric(series, errors="coerce")
        if values.dropna().empty:
            return
        display[column_name] = values
        metadata[column_name] = _metric_payload(
            column_name,
            source="Sports2D 2D TRC 关键点几何",
            unit="deg",
            is_angle=True,
            is_auxiliary=False,
            body_region_rank=BODY_RANKS.get(body_key, 110),
            side_rank=0 if side == "左侧" else 1 if side == "右侧" else 2,
            motion_rank=motion_rank,
            action_label=action_label,
            description=description,
            interpretation=(
                "该指标是单目视频平面内的二维近似，适合观察当前拍摄平面中的相对变化趋势；"
                "当存在明显斜拍、遮挡、转体或出平面运动时，不应解释为完整三维关节角。"
            ),
            camera_note=camera_note,
            raw_metric=raw_metric,
            display_metric=column_name,
            movement_plane=movement_plane,
            neutral_definition=neutral_definition,
            direction_definition=direction_definition,
            technical_source=f"Sports2D 像素 TRC：{trc_path.name}；关键点：{raw_metric}",
            is_primary_rom_metric=True,
        )

    for prefix, side in [("L", "左侧"), ("R", "右侧")]:
        side_text = side
        knee = _included_angle_2d(trc_df, f"{prefix}Hip", f"{prefix}Knee", f"{prefix}Ankle")
        add_metric(
            f"{side_text}膝关节屈曲角（2D）",
            180.0 - knee,
            side=side_text,
            body_key="knee",
            motion_rank=0,
            raw_metric=f"{prefix}Hip-{prefix}Knee-{prefix}Ankle",
            action_label=f"{side_text}膝关节屈曲/伸展活动角（2D）",
            description="由髋、膝、踝三个二维关键点计算膝关节夹角，并换算为 0°接近伸直、数值增大表示屈曲增加的活动角。",
            neutral_definition="0°近似表示视频平面内髋-膝-踝三点接近共线的膝伸直位。",
            direction_definition="数值增大表示膝关节在视频平面内屈曲增加；数值减小表示更接近伸直。",
        )

        hip = _included_angle_2d(trc_df, f"{prefix}Shoulder", f"{prefix}Hip", f"{prefix}Knee")
        add_metric(
            f"{side_text}髋关节屈曲角（2D）",
            180.0 - hip,
            side=side_text,
            body_key="hip",
            motion_rank=0,
            raw_metric=f"{prefix}Shoulder-{prefix}Hip-{prefix}Knee",
            action_label=f"{side_text}髋关节屈曲/伸展活动角（2D）",
            description="由肩、髋、膝三个二维关键点估计躯干与大腿之间的平面夹角，并换算为相对伸直/中立位的屈曲活动角。",
            neutral_definition="0°近似表示视频平面内躯干与大腿方向接近共线的髋伸直参考位。",
            direction_definition="数值增大表示髋关节在视频平面内屈曲增加；数值减小表示更接近伸展参考位。",
        )

        elbow = _included_angle_2d(trc_df, f"{prefix}Shoulder", f"{prefix}Elbow", f"{prefix}Wrist")
        add_metric(
            f"{side_text}肘关节屈曲角（2D）",
            180.0 - elbow,
            side=side_text,
            body_key="elbow",
            motion_rank=0,
            raw_metric=f"{prefix}Shoulder-{prefix}Elbow-{prefix}Wrist",
            action_label=f"{side_text}肘关节屈曲/伸展活动角（2D）",
            description="由肩、肘、腕三个二维关键点计算肘关节夹角，并换算为 0°接近伸直、数值增大表示屈曲增加的活动角。",
            neutral_definition="0°近似表示视频平面内肩-肘-腕三点接近共线的肘伸直位。",
            direction_definition="数值增大表示肘关节在视频平面内屈曲增加；数值减小表示更接近伸直。",
        )

        shoulder = _included_angle_2d(trc_df, f"{prefix}Hip", f"{prefix}Shoulder", f"{prefix}Elbow")
        add_metric(
            f"{side_text}肩关节平面抬高角（2D）",
            shoulder,
            side=side_text,
            body_key="shoulder",
            motion_rank=0,
            raw_metric=f"{prefix}Hip-{prefix}Shoulder-{prefix}Elbow",
            action_label=f"{side_text}肩关节平面抬高活动角（2D）",
            description="由髋、肩、肘三个二维关键点估计上臂相对躯干的画面内夹角，用于描述当前视频平面中的肩部抬高趋势。",
            neutral_definition="0°近似表示上臂沿躯干方向下垂或接近躯干轴线；该中立位会受躯干姿态和拍摄平面影响。",
            direction_definition="数值增大表示上臂在视频平面内相对躯干抬高增加；不能单独区分三维屈曲、外展和旋转。",
        )

        ankle = _included_angle_2d(trc_df, f"{prefix}Knee", f"{prefix}Ankle", f"{prefix}BigToe")
        add_metric(
            f"{side_text}踝关节相对中立位偏移角（2D）",
            (ankle - 90.0).abs(),
            side=side_text,
            body_key="ankle",
            motion_rank=0,
            raw_metric=f"{prefix}Knee-{prefix}Ankle-{prefix}BigToe",
            action_label=f"{side_text}踝关节相对中立位偏移角（2D）",
            description="由膝、踝、大脚趾三个二维关键点估计小腿与足部的夹角，并显示相对 90°参考位的绝对偏移量。",
            neutral_definition="0°表示视频平面内小腿与足部近似垂直于 90°参考位；这只是二维几何参考，不是临床踝中立位。",
            direction_definition="数值增大表示踝部相对参考位的偏移增大；单目二维数据无法稳定区分背屈与跖屈方向。",
        )

    if len(display.columns) <= 1:
        return None
    return {
        "df": display,
        "kind": kind,
        "metric_metadata": metadata,
        "auxiliary_stats": pd.DataFrame(),
        "source_note": f"Sports2D 像素 TRC：{trc_path.name}",
    }


def _included_angle_2d(trc_df: pd.DataFrame, proximal: str, joint: str, distal: str) -> pd.Series:
    required = [f"{marker}_{axis}" for marker in (proximal, joint, distal) for axis in ("X", "Y")]
    if any(column not in trc_df.columns for column in required):
        return pd.Series(index=trc_df.index, dtype=float)
    p1 = trc_df[[f"{proximal}_X", f"{proximal}_Y"]].astype(float).to_numpy()
    p2 = trc_df[[f"{joint}_X", f"{joint}_Y"]].astype(float).to_numpy()
    p3 = trc_df[[f"{distal}_X", f"{distal}_Y"]].astype(float).to_numpy()
    v1 = p1 - p2
    v2 = p3 - p2
    numerator = np.sum(v1 * v2, axis=1)
    denominator = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cosines = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return pd.Series(angles, index=trc_df.index)


def _find_px_trc_for_motion(path: Path) -> Path | None:
    person_match = re.search(r"person(\d+)", path.stem, re.I)
    candidates: list[Path] = []
    if person_match:
        candidates.extend(sorted(path.parent.glob(f"*px_person{person_match.group(1)}.trc")))
    candidates.extend(sorted(path.parent.glob("*_px_person*.trc")))
    return candidates[0] if candidates else None


def _unique_column_name(df: pd.DataFrame, preferred: str) -> str:
    if preferred not in df.columns:
        return preferred
    index = 2
    while f"{preferred} ({index})" in df.columns:
        index += 1
    return f"{preferred} ({index})"


def _prepare_report_video(sports_dir: Path, log: LogCallback | None) -> Path | None:
    raw_video = sports_dir / f"{sports_dir.name}.mp4"
    if not raw_video.exists():
        videos = sorted(sports_dir.glob("*.mp4"))
        raw_video = videos[0] if videos else raw_video
    if not raw_video.exists():
        return None
    return convert_video_for_browser(
        raw_video,
        sports_dir / "reports" / "assets" / f"{sports_dir.name}_browser.mp4",
        log,
    )


def _motion_payload(path: Path, df: pd.DataFrame, quality: dict[str, Any] | None = None) -> dict[str, Any]:
    safe = df.copy()
    for column in safe.columns:
        safe[column] = pd.to_numeric(safe[column], errors="coerce")
    kind = classify_motion_file(path)
    camera_note = _camera_plane_note(quality)
    kind["camera_note"] = camera_note
    standardized = _standardize_report_motion(path, safe, kind, quality)
    report_df = standardized["df"]
    report_kind = standardized["kind"]
    stats = angle_statistics(
        report_df,
        report_kind,
        quality,
        report_details=True,
        metric_metadata=standardized.get("metric_metadata"),
    )
    stats_records = _records(stats)
    auxiliary_stats = standardized.get("auxiliary_stats")
    auxiliary_stats_records = _records(auxiliary_stats) if isinstance(auxiliary_stats, pd.DataFrame) and not auxiliary_stats.empty else []
    auxiliary_stats_html = _stats_table_html(auxiliary_stats) if auxiliary_stats_records else ""
    return {
        "name": path.stem,
        "columns": list(report_df.columns),
        "stats": stats_records,
        "auxiliary_stats": auxiliary_stats_records,
        "stats_html": _stats_table_html(stats),
        "auxiliary_stats_html": auxiliary_stats_html,
        "default_metrics": _default_metrics(stats_records),
        "df": report_df,
        "camera_note": camera_note,
        "source_note": standardized.get("source_note", ""),
        **report_kind,
    }


def _motion_figure(payload: dict[str, Any]) -> go.Figure:
    df = payload["df"]
    fig = go.Figure()
    plotted_stats = [stat for stat in payload.get("stats", []) if stat.get("is_plottable")]
    for stat in plotted_stats:
        column = stat["angle"]
        if column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[column],
                mode="lines",
                name=column,
                hovertemplate="%{y:.2f}°<extra>%{fullData.name}</extra>",
                customdata=[[stat.get("movement_label", "")] for _ in range(len(df))],
            )
        )
    if not plotted_stats:
        fig.add_annotation(
            text="当前文件未包含可作为角度曲线展示的核心角度指标。",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font={"size": 14, "color": "#5f6b7a"},
        )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 48, "r": 18, "t": 28, "b": 48},
        xaxis_title="时间 (s)",
        yaxis_title="关节活动角 (deg)",
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def _stats_table_html(stats: pd.DataFrame) -> str:
    if stats.empty:
        return "<p class='muted'>没有可统计的角度数据。</p>"
    display = stats.copy()
    for col in ["min", "time_at_min", "max", "time_at_max", "mean", "std", "rom"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    headers = [
        "Metric",
        "动作含义",
        "类型",
        "Unit",
        "Min",
        "Time@Min (s)",
        "Max",
        "Time@Max (s)",
        "Mean",
        "ROM (deg) / Range",
    ]
    rows = [
        "<div class='table-wrap'><table class='stats-table'><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody>"
    ]
    for _, row in display.iterrows():
        angle = str(row["angle"])
        metric = (
            "<span class='metric-name'>"
            f"{html.escape(angle)}"
            f"<button class='icon-button metric-info' type='button' data-angle='{html.escape(angle, quote=True)}' "
            f"aria-label='查看 {html.escape(angle, quote=True)} 的指标解释'>i</button>"
            "</span>"
        )
        cells = [
            metric,
            html.escape(str(row.get("action_label", row["movement_label"]))),
            html.escape("辅助数据" if bool(row.get("is_auxiliary")) else "角度指标" if bool(row.get("is_angle", True)) else "非角度指标"),
            html.escape(str(row.get("unit", "deg"))),
            html.escape(str(row["min"])),
            html.escape(str(row["time_at_min"])),
            html.escape(str(row["max"])),
            html.escape(str(row["time_at_max"])),
            html.escape(str(row["mean"])),
            html.escape(f"{row.get('range_label', 'ROM')}: {row['rom']}"),
        ]
        rows.append(
            "<tr>"
            + "".join(
                f"<td class='metric-col'>{cell}</td>" if idx == 0 else f"<td>{cell}</td>"
                for idx, cell in enumerate(cells)
            )
            + "</tr>"
        )
    rows.append("</tbody></table></div>")
    return "".join(rows)


def _quality_status_parts(quality: dict[str, Any]) -> tuple[str, str]:
    status = quality.get("status", "unknown")
    label = {"pass": "通过", "warn": "注意", "fail": "异常", "unknown": "未评估"}.get(status, status)
    badge_class = {"pass": "good", "warn": "warn", "fail": "fail"}.get(status, "")
    return label, badge_class


def _quality_summary_html(quality: dict[str, Any]) -> str:
    label, badge_class = _quality_status_parts(quality)
    warnings = quality.get("warnings") or []
    marker_rows = quality.get("marker_error_logs") or []
    if warnings:
        summary = f"发现 {len(warnings)} 条需要复核的质量提示。请在解释角度曲线前查看完整诊断。"
    elif marker_rows:
        summary = "已读取 OpenSim marker error 日志；请结合完整诊断判断 IK/MOT 是否适合进一步解释。"
    else:
        summary = "未发现明确质量警告。仍建议先核对处理后视频中的 2D 骨架稳定性和拍摄方向。"
    return (
        "<div class='quality-summary' aria-label='质量诊断与解释边界摘要'>"
        "<div class='quality-summary-title'>"
        "<strong>质量诊断与解释边界</strong>"
        f"<span class='badge {badge_class}'>{html.escape(label)}</span>"
        "</div>"
        f"<p class='quality-summary-text'>{html.escape(summary)}</p>"
        "<button id='openQualityDetail' class='detail-button' type='button' aria-label='查看完整质量诊断与解释边界'>查看完整诊断</button>"
        "</div>"
    )


def _quality_detail_html(quality: dict[str, Any]) -> str:
    status = quality.get("status", "unknown")
    label, badge_class = _quality_status_parts(quality)
    warnings = quality.get("warnings") or ["未发现明确警告。仍建议先核对处理后视频中的 2D 骨架质量。"]
    notes = quality.get("angle_notes") or []
    marker_rows = quality.get("marker_error_logs") or []
    marker_html = ""
    if marker_rows:
        marker_html = (
            "<div class='table-wrap'><table class='quality-table'><thead><tr><th>日志</th><th>状态</th><th>帧数</th>"
            "<th>RMS 最大值 (m)</th><th>最大误差 (m)</th></tr></thead><tbody>"
        )
        for row in marker_rows:
            marker_html += (
                "<tr>"
                f"<td>{html.escape(str(row['path']))}</td>"
                f"<td>{html.escape(str(row['status']))}</td>"
                f"<td>{row['frames']}</td>"
                f"<td>{row['rms_max']:.3f}</td>"
                f"<td>{row['max_max']:.3f}</td>"
                "</tr>"
            )
        marker_html += "</tbody></table></div>"
    else:
        marker_html = (
            "<p class='muted'>未找到 OpenSim marker error。通常表示未运行 IK，"
            "或 Sports2D 未输出 OpenSim 日志；这不影响 2D 平面角报告。</p>"
        )
    card_class = "fail" if status == "fail" else "warn" if status == "warn" else "good"
    return (
        "<div class='quality'>"
        f"<div><span class='badge {badge_class}'>{html.escape(label)}</span>"
        f"<p class='note'>当前质量状态为 {html.escape(str(status))}。质量等级用于提示解释边界，不会自动证明或否定所有角度曲线；最终仍需结合处理后视频、日志和实际拍摄条件复核。</p></div>"
        "<div class='info-grid'>"
        f"<div class='info-card {card_class}'><h3>主要提示</h3><ul>"
        + "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
        + "</ul></div>"
        "<div class='info-card'><h3>解释原则</h3><ul>"
        + "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
        + "</ul></div>"
        f"<div class='info-card'><h3>OpenSim marker error</h3>{marker_html}</div>"
        "</div></div>"
    )


def _quality_dataframe(quality: dict[str, Any]) -> pd.DataFrame:
    rows = [{"type": "overall", "key": "status", "value": quality.get("status", "unknown")}]
    for warning in quality.get("warnings", []):
        rows.append({"type": "warning", "key": "warning", "value": warning})
    for note in quality.get("angle_notes", []):
        rows.append({"type": "note", "key": "angle_note", "value": note})
    for key, value in quality.get("run_log_insights", {}).items():
        rows.append({"type": "run_log", "key": key, "value": value})
    for item in quality.get("marker_error_logs", []):
        for key, value in item.items():
            rows.append({"type": "marker_error", "key": f"{item.get('path')}::{key}", "value": value})
    return pd.DataFrame(rows)


def _video_metadata_dataframe(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        original = item.get("original_metadata", {})
        prepared = item.get("prepared_metadata", {})
        rows.append(
            {
                "source_path": item.get("source_path"),
                "work_path": item.get("work_path"),
                "rotation_fixed": item.get("rotation_fixed"),
                "original_rotation": rotation_from_metadata(original),
                "prepared_rotation": rotation_from_metadata(prepared),
                "original_size": video_size_from_metadata(original),
                "prepared_size": video_size_from_metadata(prepared),
            }
        )
    return pd.DataFrame(rows)


def _video_html(video_rel: str) -> str:
    if not video_rel:
        return "<p class='muted'>未找到可播放的视频输出。</p>"
    return f'<video id="syncVideo" src="{html.escape(video_rel)}" controls preload="metadata"></video>'


def _relative_posix(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _sheet_name(name: str) -> str:
    invalid = set("[]:*?/\\")
    cleaned = "".join("_" if ch in invalid else ch for ch in name)
    return cleaned[:31] or "data"


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def _default_metrics(stats: list[dict[str, Any]]) -> list[str]:
    plottable = [item for item in stats if item.get("is_plottable", True)]
    important = [
        item["angle"]
        for item in plottable
        if any(keyword in str(item["angle"]).lower() for keyword in IMPORTANT_KEYWORDS)
    ]
    return important[:8] if important else [item["angle"] for item in plottable[:6]]


def _display_measure_name(name: str) -> str:
    text = name.replace("_", " ").strip()
    side = ""
    lower = text.lower()
    if lower.endswith(" r") or lower.endswith(" right"):
        side = "右侧 "
        text = text.rsplit(" ", 1)[0]
    elif lower.endswith(" l") or lower.endswith(" left"):
        side = "左侧 "
        text = text.rsplit(" ", 1)[0]
    replacements = {
        "Right": "右侧",
        "Left": "左侧",
        "ankle": "踝",
        "knee": "膝",
        "hip": "髋",
        "shoulder": "肩",
        "elbow": "肘",
        "wrist": "腕",
        "flexion": "屈曲",
        "adduction": "内收",
        "rotation": "旋转",
        "angle": "角度",
        "pelvis": "骨盆",
        "trunk": "躯干",
        "lumbar": "腰椎",
        "arm": "上臂",
        "forearm": "前臂",
    }
    for old, new in replacements.items():
        text = re.sub(rf"\b{old}\b", new, text, flags=re.I)
    return (side + text).strip()


def _quality_log_paths(output_dir: Path) -> list[Path]:
    paths = []
    run_log = output_dir / "run.log"
    if run_log.exists():
        paths.append(run_log)
    paths.extend(sorted(output_dir.glob("*_Sports2D/opensim_logs.txt")))
    return paths


def _marker_status(summary: dict[str, Any]) -> str:
    if summary["rms_max"] > 0.25 or summary["max_max"] > 0.50:
        return "fail"
    if summary["rms_max"] > 0.10 or summary["max_max"] > 0.25:
        return "warn"
    return "pass"


def _overall_quality_status(marker_logs: list[dict[str, Any]], warnings: list[str]) -> str:
    statuses = [item.get("status") for item in marker_logs]
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses or warnings:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "unknown"


def _first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text.replace("\n", " "))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _load_video_metadata(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "video_metadata.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _deduplicate(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
