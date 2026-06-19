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


SKELETON_EDGES = [
    ("Hip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("RAnkle", "RBigToe"),
    ("RAnkle", "RHeel"),
    ("Hip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("LAnkle", "LBigToe"),
    ("LAnkle", "LHeel"),
    ("Hip", "Neck"),
    ("Neck", "Head"),
    ("Head", "Nose"),
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
]

MARKER_ERROR_RE = re.compile(r"marker error: RMS = ([0-9.]+), max = ([0-9.]+)")
HORIZON_RE = re.compile(r"Camera horizon:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", re.I)
HEIGHT_RE = re.compile(r"person height of [0-9.]+ in meters.*?of ([0-9.]+) in pixels", re.I)
SEEN_FROM_RE = re.compile(r"Seen from the ([A-Za-z ]+)", re.I)


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


def read_trc(trc_path: Path, max_frames: int = 900) -> dict[str, Any] | None:
    lines = trc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 6:
        return None
    marker_names = [m.strip() for m in lines[3].split("\t")[2::3] if m.strip()]
    data = pd.read_csv(trc_path, sep="\t", skiprows=4)
    if data.shape[1] < 5 or not marker_names:
        return None
    frames = data.iloc[:, 0].tolist()
    times = data.iloc[:, 1].tolist()
    coord_data = data.drop(data.columns[[0, 1]], axis=1)
    coord_data = coord_data.loc[:, ~coord_data.columns.str.startswith("Unnamed")]
    if len(data) > max_frames:
        step = max(1, len(data) // max_frames)
        coord_data = coord_data.iloc[::step, :]
        frames = frames[::step]
        times = times[::step]
    marker_frames: list[dict[str, list[float | None]]] = []
    for _, row in coord_data.iterrows():
        frame: dict[str, list[float | None]] = {}
        values = row.tolist()
        for i, name in enumerate(marker_names):
            base = i * 3
            if base + 2 >= len(values):
                continue
            x = _clean_number(values[base])
            y = _clean_number(values[base + 1])
            z = _clean_number(values[base + 2])
            frame[name] = _trc_to_display_xyz(x, y, z)
        marker_frames.append(frame)
    return {
        "markers": marker_names,
        "frames": frames,
        "times": times,
        "marker_frames": marker_frames,
        "edges": [edge for edge in SKELETON_EDGES if edge[0] in marker_names and edge[1] in marker_names],
        "axis_mapping": {
            "display_x": "TRC X",
            "display_y": "TRC Z/depth",
            "display_z": "TRC Y/vertical",
            "note": "OpenSim/TRC 通常以 Y 为竖直轴；HTML 视图已映射为浏览器中的 Z 竖直轴。",
        },
    }


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
        trc_data = _find_trc_data(sports_dir)
        video = _prepare_report_video(sports_dir, log)
        write_html_report(report_path, sports_dir.name, motions, video, trc_data, quality)
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
    marker_logs = []
    insights: dict[str, Any] = {}

    kinematics = config.get("kinematics", {})
    base = config.get("base", {})
    do_ik = bool(kinematics.get("do_ik"))
    use_augmentation = bool(kinematics.get("use_augmentation"))
    if do_ik and not use_augmentation:
        warnings.append("OpenSim IK 已开启，但未启用标记增强；Sports2D 官方日志也提示这种组合容易产生很大的 IK marker error。")
    if not do_ik:
        insights["ik_status"] = "not_run"

    visible_side = [str(v).lower() for v in base.get("visible_side", [])]
    if any(side in {"front", "back"} for side in visible_side):
        warnings.append("可见侧设置为 front/back；矢状面屈伸角和单目 OpenSim IK 的可信度会明显依赖拍摄角度。")
    elif "auto" in visible_side:
        warnings.append("可见侧为 auto；请在报告中核对 Sports2D 识别的方向，必要时改为 right/left/front/back。")

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
                    f"Sports2D 自动估计的 camera horizon 为 {horizon:.2f}°；"
                    "若地面并非真的倾斜，建议检查拍摄角度或手动设置地面/标定参数。"
                )
        height_px = _first_float(HEIGHT_RE, text)
        if height_px is not None:
            insights.setdefault("person_height_px", height_px)
            if height_px < 250:
                warnings.append(f"用于尺度换算的人体高度只有 {height_px:.1f}px，尺度和 IK 结果可能不稳定。")
        seen_from = SEEN_FROM_RE.search(text)
        if seen_from:
            insights.setdefault("sports2d_seen_from", seen_from.group(1).strip())

    video_metadata = _load_video_metadata(output_dir)
    return {
        "status": _overall_quality_status(marker_logs, warnings),
        "warnings": _deduplicate(warnings),
        "marker_error_logs": marker_logs,
        "run_log_insights": insights,
        "video_metadata": video_metadata,
        "angle_notes": [
            "Sports2D 原生 MOT 角度是 2D 图像平面角：踝背屈、膝屈曲、髋屈曲、肩屈曲、肘屈曲以及节段相对水平线的角度。",
            "OpenSim IK 的 *_ik.mot 是模型坐标输出；只有 marker error 和拍摄条件通过检查时才建议用于 3D 解释。",
            "处理后视频骨架稳定说明 2D pose 检测稳定，但不自动证明单目 3D IK 或 OpenSim MOT 动作准确。",
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
    trc_data: dict[str, Any] | None,
    quality: dict[str, Any] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plotly_js = get_plotlyjs()
    motion_payload = [_motion_payload(path, df) for path, df in motions]
    figs = [_motion_figure(payload) for payload in motion_payload]
    fig_json = [json.loads(pio.to_json(fig, validate=False)) for fig in figs]
    video_rel = _relative_posix(video_path, report_path.parent) if video_path else ""
    trc_payload = trc_data or {}
    quality_payload = quality or {}
    first_table = _stats_table_html(motion_payload[0]) if motion_payload else "<p>没有角度数据。</p>"
    serializable_motion_payload = [
        {key: value for key, value in payload.items() if key != "df"} for payload in motion_payload
    ]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - Sports2D 交互报告</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fa; --panel:#ffffff; --ink:#182230; --muted:#667085; --line:#d0d7e2; --accent:#0f766e; --warn:#b45309; --bad:#b42318; --good:#047857; }}
    body {{ margin:0; font-family:"Microsoft YaHei", "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:22px 28px 14px; border-bottom:1px solid var(--line); background:var(--panel); }}
    h1 {{ margin:0 0 6px; font-size:24px; font-weight:700; }}
    h2 {{ margin:0 0 10px; font-size:18px; }}
    main {{ padding:18px 28px 32px; display:grid; grid-template-columns:minmax(420px, 1.05fr) minmax(420px, 1fr); gap:18px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }}
    .full {{ grid-column:1 / -1; }}
    video {{ width:100%; max-height:62vh; background:#111827; border-radius:6px; }}
    .plot {{ width:100%; height:520px; }}
    .plot.small {{ height:500px; }}
    .tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
    .tabs button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; }}
    .tabs button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:first-child, td:first-child {{ text-align:left; }}
    .muted {{ color:var(--muted); font-size:13px; line-height:1.55; }}
    .quality {{ display:grid; grid-template-columns:220px minmax(0, 1fr); gap:12px; align-items:start; }}
    .badge {{ display:inline-block; border-radius:999px; padding:5px 10px; color:#fff; background:var(--accent); font-weight:700; }}
    .badge.good {{ background:var(--good); }}
    .badge.warn {{ background:var(--warn); }}
    .badge.fail {{ background:var(--bad); }}
    .warning-list {{ margin:0; padding-left:18px; line-height:1.6; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns:1fr; padding:12px; }} .quality {{ grid-template-columns:1fr; }} }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header>
    <h1>{html.escape(title)} - Sports2D 交互报告</h1>
    <div class="muted">曲线使用统一悬停模式；鼠标停在某一时刻时会显示各列数值，并同步视频和三维标记视图。报告会区分 2D 视频平面角与 OpenSim IK 坐标。</div>
  </header>
  <main>
    <section>
      <h2>视频</h2>
      {_video_html(video_rel)}
    </section>
    <section>
      <h2>角度/坐标曲线</h2>
      <div id="motionTabs" class="tabs"></div>
      <div id="anglePlot" class="plot"></div>
    </section>
    <section class="full">
      <h2>质量诊断</h2>
      {_quality_html(quality_payload)}
    </section>
    <section class="full">
      <h2>三维标记视图</h2>
      <div id="markerPlot" class="plot small"></div>
      <p class="muted">此视图用于检查 TRC 标记轨迹方向，不等同于 OpenSim 模型可视化。OpenSim/TRC 通常以 Y 为竖直轴，本报告已将 TRC Y 映射为浏览器中的竖直轴。</p>
    </section>
    <section class="full">
      <h2>统计</h2>
      <div id="statsTable">{first_table}</div>
    </section>
  </main>
  <script>
    const motionPayload = {json.dumps(serializable_motion_payload, ensure_ascii=False)};
    const figPayload = {json.dumps(fig_json, ensure_ascii=False)};
    const trcPayload = {json.dumps(trc_payload, ensure_ascii=False)};
    const video = document.getElementById('syncVideo');
    const tabs = document.getElementById('motionTabs');
    const plot = document.getElementById('anglePlot');
    const stats = document.getElementById('statsTable');

    function renderMotion(index) {{
      Plotly.react(plot, figPayload[index].data, figPayload[index].layout, {{responsive:true}});
      stats.innerHTML = motionPayload[index].stats_html;
      [...tabs.children].forEach((button, i) => button.classList.toggle('active', i === index));
      plot.on('plotly_hover', event => {{
        const x = event.points && event.points[0] ? Number(event.points[0].x) : NaN;
        if (video && Number.isFinite(x)) video.currentTime = Math.max(0, x);
        updateMarkerByTime(x);
      }});
    }}

    motionPayload.forEach((item, index) => {{
      const button = document.createElement('button');
      button.textContent = item.name + ' · ' + item.kind_short;
      button.title = item.kind_note;
      button.onclick = () => renderMotion(index);
      tabs.appendChild(button);
    }});
    if (figPayload.length) renderMotion(0);

    function markerFrameToTrace(frame) {{
      const xs = [], ys = [], zs = [], labels = [];
      Object.entries(frame || {{}}).forEach(([name, xyz]) => {{
        if (!xyz || xyz.some(v => v === null || Number.isNaN(Number(v)))) return;
        xs.push(xyz[0]); ys.push(xyz[1]); zs.push(xyz[2]); labels.push(name);
      }});
      const edgeX = [], edgeY = [], edgeZ = [];
      (trcPayload.edges || []).forEach(edge => {{
        const a = frame[edge[0]], b = frame[edge[1]];
        if (!a || !b) return;
        edgeX.push(a[0], b[0], null); edgeY.push(a[1], b[1], null); edgeZ.push(a[2], b[2], null);
      }});
      return [
        {{type:'scatter3d', mode:'lines', x:edgeX, y:edgeY, z:edgeZ, line:{{color:'#2563eb', width:5}}, hoverinfo:'skip', name:'骨架'}},
        {{type:'scatter3d', mode:'markers+text', x:xs, y:ys, z:zs, text:labels, textposition:'top center', marker:{{size:4, color:'#dc2626'}}, name:'标记点'}}
      ];
    }}

    function updateMarkerByTime(timeValue) {{
      if (!trcPayload.marker_frames || !trcPayload.marker_frames.length) return;
      const times = trcPayload.times || [];
      let best = 0, bestDistance = Infinity;
      times.forEach((t, i) => {{
        const d = Math.abs(Number(t) - Number(timeValue));
        if (d < bestDistance) {{ bestDistance = d; best = i; }}
      }});
      Plotly.react('markerPlot', markerFrameToTrace(trcPayload.marker_frames[best]), {{
        margin:{{l:0,r:0,t:18,b:0}},
        scene:{{
          aspectmode:'data',
          xaxis:{{title:'X (TRC X)'}},
          yaxis:{{title:'深度/平面 (TRC Z)'}},
          zaxis:{{title:'竖直 (TRC Y)'}},
          camera:{{eye:{{x:1.7,y:-2.0,z:1.1}}, up:{{x:0,y:0,z:1}}}}
        }},
        showlegend:false
      }}, {{responsive:true}});
    }}
    if (trcPayload.marker_frames && trcPayload.marker_frames.length) {{
      updateMarkerByTime(trcPayload.times ? trcPayload.times[0] : 0);
    }} else {{
      document.getElementById('markerPlot').innerHTML = '<p class="muted">未找到可用于三维视图的米制 TRC 文件。</p>';
    }}
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
            stats = angle_statistics(df)
            stats.insert(0, "source", str(path.relative_to(output_dir)))
            stats.insert(1, "data_kind", classify_motion_file(path)["kind_short"])
            summary_rows.append(stats)
            data_sheet = _sheet_name(path.stem)
            df.to_excel(writer, sheet_name=data_sheet, index=False)
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


def angle_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "time" not in df.columns:
        return pd.DataFrame(rows)
    for column in df.columns:
        if column == "time":
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if series.dropna().empty:
            continue
        min_idx = series.idxmin()
        max_idx = series.idxmax()
        rows.append(
            {
                "angle": column,
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
        "kind_note": "这是由视频平面关键点计算的 2D 角度，不是完整 3D 屈曲/外展/旋转坐标。",
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


def _find_trc_data(sports_dir: Path) -> dict[str, Any] | None:
    for pattern in ["*_m_*person*.trc", "*_m_person*.trc", "*_m*.trc", "*_px_person*.trc"]:
        for trc_path in sorted(sports_dir.rglob(pattern)):
            data = read_trc(trc_path)
            if data:
                return data
    return None


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
    stats_html = _stats_table_html({"df": safe})
    kind = classify_motion_file(path)
    return {
        "name": path.stem,
        "columns": list(safe.columns),
        "records": safe.astype(object).where(pd.notnull(safe), None).to_dict(orient="records"),
        "stats_html": stats_html,
        "df": safe,
        **kind,
    }


def _motion_figure(payload: dict[str, Any]) -> go.Figure:
    df = payload["df"]
    fig = go.Figure()
    for column in df.columns:
        if column == "time":
            continue
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[column],
                mode="lines",
                name=column,
                hovertemplate="%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 48, "r": 18, "t": 24, "b": 48},
        xaxis_title="时间 (s)",
        yaxis_title=f"{payload['kind']} (deg 或模型单位)",
        legend={"orientation": "h", "y": -0.22},
        annotations=[
            {
                "text": payload["kind_note"],
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12, "color": "#667085"},
            }
        ],
    )
    return fig


def _stats_table_html(payload: dict[str, Any]) -> str:
    stats = angle_statistics(payload["df"])
    if stats.empty:
        return "<p class='muted'>没有可统计的角度数据。</p>"
    display = stats.copy()
    for col in ["min", "time_at_min", "max", "time_at_max", "mean", "std", "rom"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    headers = ["角度/坐标", "最小值", "最小值时间", "最大值", "最大值时间", "均值", "标准差", "ROM"]
    columns = ["angle", "min", "time_at_min", "max", "time_at_max", "mean", "std", "rom"]
    rows = ["<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"]
    for _, row in display.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in columns) + "</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def _quality_html(quality: dict[str, Any]) -> str:
    status = quality.get("status", "unknown")
    label = {"pass": "通过", "warn": "注意", "fail": "异常", "unknown": "未评估"}.get(status, status)
    badge_class = {"pass": "good", "warn": "warn", "fail": "fail"}.get(status, "")
    warnings = quality.get("warnings") or ["未发现明确警告。"]
    notes = quality.get("angle_notes") or []
    marker_rows = quality.get("marker_error_logs") or []
    marker_html = ""
    if marker_rows:
        marker_html = "<table><thead><tr><th>日志</th><th>状态</th><th>帧数</th><th>RMS最大(m)</th><th>最大误差(m)</th></tr></thead><tbody>"
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
        marker_html = "<p class='muted'>未找到 OpenSim marker error；可能未运行 IK，或 Sports2D 未输出 OpenSim 日志。</p>"
    return (
        "<div class='quality'>"
        f"<div><span class='badge {badge_class}'>{html.escape(label)}</span></div>"
        "<div>"
        "<ul class='warning-list'>"
        + "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
        + "</ul>"
        + marker_html
        + "<p class='muted'>"
        + "<br>".join(html.escape(str(note)) for note in notes)
        + "</p>"
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


def _clean_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _trc_to_display_xyz(x: float | None, y: float | None, z: float | None) -> list[float | None]:
    # OpenSim/TRC uses Y as vertical. Plotly's intuitive vertical screen axis is Z.
    return [x, z, y]


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
