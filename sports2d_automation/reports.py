from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .config import flatten_dict
from .video import convert_video_for_browser


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
            frame[name] = [_clean_number(values[base]), _clean_number(values[base + 1]), _clean_number(values[base + 2])]
        marker_frames.append(frame)
    return {
        "markers": marker_names,
        "frames": frames,
        "times": times,
        "marker_frames": marker_frames,
        "edges": [edge for edge in SKELETON_EDGES if edge[0] in marker_names and edge[1] in marker_names],
    }


def generate_reports_for_job(
    output_dir: Path,
    config: dict[str, Any],
    environment_report: dict[str, Any],
    log: LogCallback | None = None,
) -> tuple[list[Path], list[Path]]:
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
        write_html_report(report_path, sports_dir.name, motions, video, trc_data)
        html_reports.append(report_path)
        if log:
            log(f"HTML 交互报告已生成：{report_path}")

    excel_reports: list[Path] = []
    if all_motions:
        excel_path = output_dir / "analysis_report.xlsx"
        write_excel_report(excel_path, all_motions, config, environment_report, output_dir)
        excel_reports.append(excel_path)
        if log:
            log(f"Excel 报告已生成：{excel_path}")
    return html_reports, excel_reports


def write_html_report(
    report_path: Path,
    title: str,
    motions: list[tuple[Path, pd.DataFrame]],
    video_path: Path | None,
    trc_data: dict[str, Any] | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plotly_js = get_plotlyjs()
    motion_payload = [_motion_payload(path, df) for path, df in motions]
    figs = [_motion_figure(payload) for payload in motion_payload]
    fig_json = [json.loads(pio.to_json(fig, validate=False)) for fig in figs]
    video_rel = _relative_posix(video_path, report_path.parent) if video_path else ""
    trc_payload = trc_data or {}
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
    :root {{ color-scheme: light; --bg:#f6f7f9; --panel:#ffffff; --ink:#17202a; --muted:#667085; --line:#d7dde5; --accent:#0f766e; }}
    body {{ margin:0; font-family:"Microsoft YaHei", "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:22px 28px 14px; border-bottom:1px solid var(--line); background:var(--panel); }}
    h1 {{ margin:0; font-size:24px; font-weight:700; }}
    main {{ padding:18px 28px 32px; display:grid; grid-template-columns:minmax(420px, 1.1fr) minmax(420px, 1fr); gap:18px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }}
    .full {{ grid-column:1 / -1; }}
    video {{ width:100%; max-height:62vh; background:#111827; border-radius:6px; }}
    .plot {{ width:100%; height:520px; }}
    .plot.small {{ height:460px; }}
    .tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
    .tabs button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; }}
    .tabs button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }}
    th:first-child, td:first-child {{ text-align:left; }}
    .muted {{ color:var(--muted); font-size:13px; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns:1fr; padding:12px; }} }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header>
    <h1>{html.escape(title)} - Sports2D 交互报告</h1>
    <div class="muted">曲线使用统一悬停模式；鼠标停在某一时刻时会显示所有关节/节段角度，并同步视频时间。</div>
  </header>
  <main>
    <section>
      <h2>视频</h2>
      {_video_html(video_rel)}
    </section>
    <section>
      <h2>角度曲线</h2>
      <div id="motionTabs" class="tabs"></div>
      <div id="anglePlot" class="plot"></div>
    </section>
    <section class="full">
      <h2>三维标记视图</h2>
      <div id="markerPlot" class="plot small"></div>
      <p class="muted">如果没有米制 TRC 文件，本区域会显示为空提示。三维视图用于快速检查轨迹方向，不替代 OpenSim 检查。</p>
    </section>
    <section class="full">
      <h2>角度统计</h2>
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
      button.textContent = item.name;
      button.onclick = () => renderMotion(index);
      tabs.appendChild(button);
    }});
    if (figPayload.length) renderMotion(0);

    function markerFrameToTrace(frame) {{
      const xs = [], ys = [], zs = [], labels = [];
      Object.entries(frame || {{}}).forEach(([name, xyz]) => {{
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
        scene:{{aspectmode:'data', xaxis:{{title:'X'}}, yaxis:{{title:'Y'}}, zaxis:{{title:'Z'}}}},
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
) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_rows = []
        for path, df in motions:
            stats = angle_statistics(df)
            stats.insert(0, "source", str(path.relative_to(output_dir)))
            summary_rows.append(stats)
            data_sheet = _sheet_name(path.stem)
            df.to_excel(writer, sheet_name=data_sheet, index=False)
        if summary_rows:
            pd.concat(summary_rows, ignore_index=True).to_excel(writer, sheet_name="角度统计", index=False)
        pd.DataFrame(
            [{"key": k, "value": str(v)} for k, v in flatten_dict(config).items()]
        ).to_excel(writer, sheet_name="参数", index=False)
        pd.DataFrame(
            [{"key": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
             for k, v in environment_report.items()]
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


def _load_motion_files(sports_dir: Path) -> list[tuple[Path, pd.DataFrame]]:
    motions = []
    for mot_path in sorted(sports_dir.glob("*.mot")):
        try:
            df = read_mot(mot_path)
        except Exception:
            continue
        if "time" in df.columns:
            motions.append((mot_path, df))
    return motions


def _find_trc_data(sports_dir: Path) -> dict[str, Any] | None:
    for pattern in ["*_m_*person*.trc", "*_m_person*.trc", "*_m*.trc"]:
        for trc_path in sorted(sports_dir.glob(pattern)):
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
    return {
        "name": path.stem,
        "columns": list(safe.columns),
        "records": safe.astype(object).where(pd.notnull(safe), None).to_dict(orient="records"),
        "stats_html": stats_html,
        "df": safe,
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
                hovertemplate="%{y:.2f}°<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 48, "r": 18, "t": 24, "b": 48},
        xaxis_title="时间 (s)",
        yaxis_title="角度 (deg)",
        legend={"orientation": "h", "y": -0.22},
    )
    return fig


def _stats_table_html(payload: dict[str, Any]) -> str:
    stats = angle_statistics(payload["df"])
    if stats.empty:
        return "<p class='muted'>没有可统计的角度数据。</p>"
    display = stats.copy()
    for col in ["min", "time_at_min", "max", "time_at_max", "mean", "std", "rom"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    headers = ["角度", "最小值", "最小值时间", "最大值", "最大值时间", "均值", "标准差", "ROM"]
    columns = ["angle", "min", "time_at_min", "max", "time_at_max", "mean", "std", "rom"]
    rows = ["<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"]
    for _, row in display.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in columns) + "</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


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
