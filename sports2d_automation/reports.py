from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Callable

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
)


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
            "本报告的核心结果是 Sports2D 原生 2D 视频平面角。它反映关键点在视频平面中的几何关系，不等同于三维解剖角。",
            "当摄像机近似垂直于动作平面时，膝、髋、踝等 2D 角度可用于描述该平面内的屈伸趋势；摄像机为正面、背面或明显斜拍时，应解释为视频平面角，而不是矢状面角。",
            "OpenSim *_ik.mot 若存在，会作为附加统计展示。只有 marker error、拍摄方向、尺度和标定均通过检查时，才建议进一步用于三维解释。",
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
    motion_payload = [_motion_payload(path, df) for path, df in motions]
    figs = [_motion_figure(payload) for payload in motion_payload]
    fig_json = [json.loads(pio.to_json(fig, validate=False)) for fig in figs]
    video_rel = _relative_posix(video_path, report_path.parent) if video_path else ""
    quality_payload = quality or {}
    serializable_motion_payload = [
        {key: value for key, value in payload.items() if key != "df"} for payload in motion_payload
    ]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - Sports2D 运动学报告</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fa; --panel:#ffffff; --ink:#17202a; --muted:#5f6b7a; --line:#d7dee8; --accent:#0f766e; --accent-soft:#e6f4f1; --warn:#a15c07; --warn-soft:#fff4df; --bad:#b42318; --bad-soft:#fff0ee; --good:#047857; --good-soft:#e8f7ef; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei", "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--ink); line-height:1.55; }}
    header {{ padding:22px 28px 16px; border-bottom:1px solid var(--line); background:var(--panel); }}
    h1 {{ margin:0 0 6px; font-size:24px; font-weight:700; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:18px; font-weight:700; letter-spacing:0; }}
    h3 {{ margin:0 0 8px; font-size:15px; font-weight:700; letter-spacing:0; }}
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
    .badge {{ display:inline-block; border-radius:999px; padding:5px 10px; color:#fff; background:var(--muted); font-weight:700; }}
    .badge.good {{ background:var(--good); }}
    .badge.warn {{ background:var(--warn); }}
    .badge.fail {{ background:var(--bad); }}
    .info-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }}
    .info-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
    .info-card.good {{ background:var(--good-soft); border-color:#b7e4cc; }}
    .info-card.warn {{ background:var(--warn-soft); border-color:#f0d49a; }}
    .info-card.fail {{ background:var(--bad-soft); border-color:#f4b6ad; }}
    ul {{ margin:0; padding-left:18px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child {{ text-align:left; }}
    .definition-list {{ display:grid; gap:8px; }}
    .definition-item {{ border-bottom:1px solid var(--line); padding:8px 0; }}
    .definition-item:last-child {{ border-bottom:0; }}
    .definition-item strong {{ display:block; margin-bottom:3px; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns:1fr; padding:12px; }} .quality {{ grid-template-columns:1fr; }} .plot {{ height:460px; }} }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header>
    <h1>{html.escape(title)} - Sports2D 运动学报告</h1>
    <div class="muted">本报告以二维视频平面角为核心输出。所有角度解释均围绕 Sports2D 的关键点几何定义展开；不再展示三维骨架模型，以避免把单目估计误解为可靠三维运动。</div>
  </header>
  <main>
    <section>
      <h2>视频核对</h2>
      {_video_html(video_rel)}
      <p class="note">请先确认处理后视频中的 2D 骨架是否稳定、是否跟随同一名受试者、左右方向是否符合实际。若视频叠加骨架已经错误，角度曲线不应继续用于结论。</p>
    </section>
    <section>
      <h2>角度曲线</h2>
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
      <h2>质量诊断与解释边界</h2>
      {_quality_html(quality_payload)}
    </section>
    <section class="full">
      <h2>角度定义与动作含义</h2>
      <div id="definitionList" class="definition-list"></div>
    </section>
    <section class="full">
      <h2>完整统计表</h2>
      <div id="statsTable"></div>
    </section>
  </main>
  <script>
    const motionPayload = {json.dumps(serializable_motion_payload, ensure_ascii=False)};
    const figPayload = {json.dumps(fig_json, ensure_ascii=False)};
    const video = document.getElementById('syncVideo');
    const tabs = document.getElementById('motionTabs');
    const plot = document.getElementById('anglePlot');
    const controls = document.getElementById('metricControls');
    const cards = document.getElementById('metricCards');
    const statsTable = document.getElementById('statsTable');
    const definitions = document.getElementById('definitionList');
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
      renderDefinitions(index);
      statsTable.innerHTML = motionPayload[index].stats_html;
      plot.on('plotly_hover', event => {{
        const x = event.points && event.points[0] ? Number(event.points[0].x) : NaN;
        if (video && Number.isFinite(x)) video.currentTime = Math.max(0, x);
      }});
    }}

    function renderMetricControls(index) {{
      const item = motionPayload[index];
      const defaults = new Set(item.default_metrics || []);
      controls.innerHTML = '';
      item.stats.forEach(stat => {{
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
        label.appendChild(document.createTextNode(stat.display_name || stat.angle));
        controls.appendChild(label);
      }});
      applyMetricSelection(index);
    }}

    function applyMetricSelection(index) {{
      const selected = new Set(selectedNames());
      const traces = figPayload[index].data || [];
      const visible = traces.map(trace => selected.size === 0 || selected.has(trace.name) ? true : 'legendonly');
      if (visible.length) Plotly.restyle(plot, 'visible', visible);
    }}

    function renderMetricCards(index) {{
      const selected = new Set(selectedNames());
      const rows = motionPayload[index].stats.filter(stat => selected.size === 0 || selected.has(stat.angle));
      if (!rows.length) {{
        cards.innerHTML = '<p class="muted">请至少选择一个关节或节段。</p>';
        return;
      }}
      cards.innerHTML = rows.map(stat => `
        <article class="metric-card">
          <div class="label">${{escapeHtml(stat.kind_short)}} · ${{escapeHtml(stat.movement_label)}}</div>
          <div class="value">${{fmt(stat.rom)}}° ROM</div>
          <h3>${{escapeHtml(stat.display_name || stat.angle)}}</h3>
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

    function renderDefinitions(index) {{
      const rows = motionPayload[index].stats;
      definitions.innerHTML = rows.map(stat => `
        <div class="definition-item">
          <strong>${{escapeHtml(stat.display_name || stat.angle)}}：${{escapeHtml(stat.movement_label)}}</strong>
          <div class="muted">${{escapeHtml(stat.description)}} ${{escapeHtml(stat.interpretation)}}</div>
        </div>
      `).join('');
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
      if (button) seekTo(button.dataset.time);
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


def angle_statistics(df: pd.DataFrame, kind: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    if "time" not in df.columns:
        return pd.DataFrame(rows)
    kind = kind or {"kind_short": "数据", "kind": "运动数据", "kind_note": ""}
    for column in df.columns:
        if column == "time":
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if series.dropna().empty:
            continue
        min_idx = series.idxmin()
        max_idx = series.idxmax()
        meta = measure_metadata(column, kind)
        rows.append(
            {
                "angle": column,
                "display_name": meta["display_name"],
                "kind_short": kind["kind_short"],
                "movement_label": meta["movement_label"],
                "description": meta["description"],
                "interpretation": meta["interpretation"],
                "min": float(series.min()),
                "time_at_min": float(df.loc[min_idx, "time"]),
                "max": float(series.max()),
                "time_at_max": float(df.loc[max_idx, "time"]),
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series.dropna()) > 1 else 0.0,
                "rom": float(series.max() - series.min()),
            }
        )
    return pd.DataFrame(rows)


def classify_motion_file(path: Path) -> dict[str, str]:
    stem = path.stem.lower()
    if stem.endswith("_ik") or "_ik" in stem:
        return {
            "kind_short": "OpenSim IK",
            "kind": "OpenSim IK 模型坐标",
            "kind_note": "这是 OpenSim 逆运动学输出的模型坐标；请结合 marker error 和拍摄条件判断可信度。",
        }
    return {
        "kind_short": "2D平面角",
        "kind": "Sports2D 2D 视频平面角",
        "kind_note": "这是由视频平面关键点计算的 2D 角度，不是完整三维解剖角。",
    }


def measure_metadata(name: str, kind: dict[str, str]) -> dict[str, str]:
    display = _display_measure_name(name)
    lower = name.lower()
    if kind["kind_short"] == "OpenSim IK":
        return _opensim_measure_metadata(name, display, lower)
    return _sports2d_measure_metadata(name, display, lower)


def _sports2d_measure_metadata(name: str, display: str, lower: str) -> dict[str, str]:
    side = "右侧" if "right" in lower else "左侧" if "left" in lower else "中线/整体"
    if "ankle" in lower:
        label = f"{side}踝背屈/跖屈趋势"
        desc = "Sports2D 用足跟、大脚趾、踝和膝在视频平面中的位置计算踝关节角。"
        interp = "侧面拍摄时主要反映踝背屈与跖屈趋势；正面或斜向拍摄时只能解释为视频平面内角度。"
    elif "knee" in lower:
        label = f"{side}膝屈曲/伸展趋势"
        desc = "该角度由髋、膝、踝三个关键点在视频平面中的夹角计算。"
        interp = "数值变化可描述膝关节在拍摄平面内的屈伸过程；摄像机不在矢状面时不要直接当作矢状面膝屈曲角。"
    elif "hip" in lower:
        label = f"{side}髋屈曲/伸展趋势"
        desc = "该角度由膝、髋、肩三个关键点在视频平面中的夹角计算。"
        interp = "侧面拍摄时可近似描述髋屈伸趋势；正面拍摄时会混入外展、躯干侧倾和透视影响。"
    elif "shoulder" in lower:
        label = f"{side}肩屈曲/伸展趋势"
        desc = "该角度由髋、肩、肘三个关键点在视频平面中的夹角计算。"
        interp = "可用于观察上臂相对躯干的平面内运动；它不是肩关节完整三维屈曲、外展或旋转分解。"
    elif "elbow" in lower:
        label = f"{side}肘屈曲/伸展趋势"
        desc = "该角度由腕、肘、肩三个关键点在视频平面中的夹角计算。"
        interp = "主要反映肘关节在画面平面内的屈伸趋势，受前臂旋前旋后和透视影响。"
    elif "wrist" in lower:
        label = f"{side}腕部平面角"
        desc = "该角度来自腕部相邻关键点在视频平面中的几何关系。"
        interp = "腕部关键点更容易受遮挡和检测噪声影响，应优先结合处理后视频核对。"
    elif any(segment in lower for segment in ["foot", "shank", "thigh", "pelvis", "trunk", "head", "arm", "forearm"]):
        label = "节段相对水平线角度"
        desc = "节段角表示对应身体节段在视频平面中相对水平线的逆时针角度。"
        interp = "它描述的是节段姿态，不是关节夹角；摄像机倾斜时需关注地面角校正是否合理。"
    else:
        label = "视频平面角"
        desc = "该列来自 Sports2D MOT 文件，表示视频平面中的几何角度。"
        interp = "请结合列名、拍摄方向和处理后视频解释，不应自动视为三维解剖角。"
    return {
        "display_name": display,
        "movement_label": label,
        "description": desc,
        "interpretation": interp,
    }


def _opensim_measure_metadata(name: str, display: str, lower: str) -> dict[str, str]:
    if "hip_flexion" in lower:
        label = "髋屈曲/伸展坐标"
    elif "hip_adduction" in lower:
        label = "髋内收/外展坐标"
    elif "hip_rotation" in lower:
        label = "髋内旋/外旋坐标"
    elif "knee" in lower:
        label = "膝屈曲/伸展坐标"
    elif "ankle" in lower:
        label = "踝背屈/跖屈坐标"
    elif "pelvis" in lower:
        label = "骨盆姿态坐标"
    elif "lumbar" in lower:
        label = "腰椎/躯干姿态坐标"
    elif "arm_flex" in lower:
        label = "肩屈曲/伸展坐标"
    elif "arm_add" in lower:
        label = "肩内收/外展坐标"
    elif "arm_rot" in lower:
        label = "肩旋转坐标"
    elif "elbow" in lower:
        label = "肘屈曲/伸展坐标"
    elif "wrist" in lower:
        label = "腕关节坐标"
    else:
        label = "OpenSim 模型坐标"
    return {
        "display_name": display,
        "movement_label": label,
        "description": "该列来自 OpenSim 逆运动学 MOT 文件，是模型坐标而不是 Sports2D 原生 2D 角度。",
        "interpretation": "只有当 marker error、尺度、拍摄方向和标定均合理时，才建议进一步按三维模型坐标解释。",
    }


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


def _motion_payload(path: Path, df: pd.DataFrame) -> dict[str, Any]:
    safe = df.copy()
    for column in safe.columns:
        safe[column] = pd.to_numeric(safe[column], errors="coerce")
    kind = classify_motion_file(path)
    stats = angle_statistics(safe, kind)
    stats_records = _records(stats)
    return {
        "name": path.stem,
        "columns": list(safe.columns),
        "stats": stats_records,
        "stats_html": _stats_table_html(stats),
        "default_metrics": _default_metrics(stats_records),
        "df": safe,
        **kind,
    }


def _motion_figure(payload: dict[str, Any]) -> go.Figure:
    df = payload["df"]
    fig = go.Figure()
    for column in df.columns:
        if column == "time":
            continue
        meta = measure_metadata(column, payload)
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[column],
                mode="lines",
                name=column,
                hovertemplate="%{y:.2f}°<extra>%{fullData.name}</extra>",
                customdata=[[meta["movement_label"]] for _ in range(len(df))],
            )
        )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 48, "r": 18, "t": 28, "b": 48},
        xaxis_title="时间 (s)",
        yaxis_title=f"{payload['kind']} (deg)",
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def _stats_table_html(stats: pd.DataFrame) -> str:
    if stats.empty:
        return "<p class='muted'>没有可统计的角度数据。</p>"
    display = stats.copy()
    for col in ["min", "time_at_min", "max", "time_at_max", "mean", "std", "rom"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    headers = ["指标", "动作含义", "最小值", "最小值时间", "最大值", "最大值时间", "均值", "ROM"]
    columns = [
        "display_name",
        "movement_label",
        "min",
        "time_at_min",
        "max",
        "time_at_max",
        "mean",
        "rom",
    ]
    rows = ["<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"]
    for _, row in display.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in columns) + "</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def _quality_html(quality: dict[str, Any]) -> str:
    status = quality.get("status", "unknown")
    label = {"pass": "通过", "warn": "注意", "fail": "异常", "unknown": "未评估"}.get(status, status)
    badge_class = {"pass": "good", "warn": "warn", "fail": "fail"}.get(status, "")
    warnings = quality.get("warnings") or ["未发现明确警告。仍建议先核对处理后视频中的 2D 骨架质量。"]
    notes = quality.get("angle_notes") or []
    marker_rows = quality.get("marker_error_logs") or []
    marker_html = ""
    if marker_rows:
        marker_html = (
            "<table><thead><tr><th>日志</th><th>状态</th><th>帧数</th>"
            "<th>RMS最大(m)</th><th>最大误差(m)</th></tr></thead><tbody>"
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
        marker_html += "</tbody></table>"
    else:
        marker_html = (
            "<p class='muted'>未找到 OpenSim marker error。通常表示未运行 IK，"
            "或 Sports2D 未输出 OpenSim 日志；这不影响 2D 平面角报告。</p>"
        )
    card_class = "fail" if status == "fail" else "warn" if status == "warn" else "good"
    return (
        "<div class='quality'>"
        f"<div><span class='badge {badge_class}'>{html.escape(label)}</span>"
        "<p class='note'>质量等级用于提醒解释边界，不会自动证明或否定所有曲线。</p></div>"
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
    important = [
        item["angle"]
        for item in stats
        if any(keyword in str(item["angle"]).lower() for keyword in IMPORTANT_KEYWORDS)
    ]
    return important[:8] if important else [item["angle"] for item in stats[:6]]


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
