# Sports2D 自动化中文桌面工具

这个项目为 Sports2D 提供中文桌面 GUI、批处理流水线、HTML 交互报告和 Excel 汇总报告。默认环境为：

```text
D:\Application\Anaconda\envs\sports3d
```

双击 `run_gui.bat` 启动；也可以运行：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.gui
```

## 基本流程

1. 在 `Inputs` 下为每个分析任务新建一个文件夹，把视频放进去。
2. 启动 GUI，左侧勾选一个或多个输入作业。
3. 选择分析预设：
   - `推荐新手模式`：默认模式，生成 2D 角度、处理后视频、TRC/MOT、HTML 和 Excel，不默认运行 OpenSim IK。
   - `完整 OpenSim 模式`：在推荐输出基础上运行 IK，并默认启用标记增强。
   - `专家模式`：显示底层参数，允许关闭标记增强等高风险组合。
4. 填写身高、体重、可见侧、时间范围等必要信息。
5. 点击“运行所选分析”。

## 重要原则

- 处理后视频骨架稳定，说明 2D pose 检测稳定；它不能单独证明 OpenSim IK 或 3D MOT 动作准确。
- Sports2D 原生角度是 2D 视频平面角，不是完整 3D 屈曲、外展、旋转角。
- OpenSim `*_ik.mot` 是模型坐标输出，必须结合 HTML/Excel 中的 marker error、相机方向、地面角和拍摄条件判断可信度。
- 普通视频的宽高会自动读取；GUI 不要求新手手动填写输入宽度/高度。那些参数只保留在专家模式中，用于 webcam 或特殊输入。
- 运行 IK 时建议启用标记增强。关闭标记增强可能产生非常大的 marker error 和严重扭曲的 MOT/OpenSim 动作。
- `脚贴地修正` 只适合双脚基本始终贴地的动作；举重、跑跳或脚跟明显离地时不要默认开启。

## 输出结构

每次运行都会创建独立目录，避免旧结果和新日志混在一起：

```text
Outputs/
  <ASCII作业名>/
    latest_run.json
    runs/
      20260620_153012_123456/
        run_config.toml
        run_status.json
        run.log
        environment_report.json
        video_metadata.json
        _work/
        <video>_Sports2D/
        analysis_report.xlsx
```

`run_status.json` 会记录 `running / success / failed / canceled`。取消或失败的运行不会展示旧报告作为本次结果。

## 报告内容

- HTML 交互报告：
  - 视频和角度曲线同步。
  - 鼠标悬停显示当前时刻各角度/坐标值。
  - 明确区分 `Sports2D 2D 视频平面角` 与 `OpenSim IK 模型坐标`。
  - 三维标记视图已把 TRC/OpenSim 的 `Y` 轴映射为浏览器中的竖直轴。
  - 显示 OpenSim marker error、相机 horizon、可见侧等质量诊断。
- Excel 报告：
  - 原始角度/坐标数据。
  - 最小值、最大值、均值、标准差、ROM 和峰值时间。
  - 参数、环境、视频元数据、质量诊断、输出文件索引。

## 视频方向处理

工具会先用 `ffprobe` 检查视频旋转元数据。若发现 `rotation`，会用 `ffmpeg` 生成已物理旋转且清除旋转元数据的临时 MP4，再交给 Sports2D。这样可以避免竖屏视频在 OpenCV/Sports2D/OpenSim 链路中被当作横屏处理。

## 命令行

列出输入作业：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli list
```

运行推荐模式：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli run --job OfficialDemo
```

运行完整 OpenSim 模式：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli run --job OfficialDemo --ik
```

快速短片段检查：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli run --job OfficialDemo --quick --start 0 --end 1
```

检查环境：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli check-env
```

## 环境与更新

GUI 中的“检查环境”会检查 Sports2D、Pose2Sim、OpenSim、PySide6、Plotly、OpenPyXL、ffmpeg、ffprobe、ONNXRuntime providers 和 DeepSort 依赖。

“一键更新 Sports2D/Pose2Sim”只运行：

```powershell
python -m pip install -U sports2d pose2sim
```

它不会自动升级 OpenSim。

## Git 注意

真实视频和分析输出默认不提交到 GitHub。仓库只保留代码、脚本、说明和 `Inputs` / `Outputs` 的占位文件。
