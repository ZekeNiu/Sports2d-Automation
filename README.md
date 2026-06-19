# Sports2D 自动化中文桌面工具

这个项目为 Sports2D 提供一个中文桌面图形界面和批处理流水线。它会从 `Inputs` 子文件夹读取视频，把 Sports2D 原生结果、日志、HTML 交互报告和 Excel 汇总报告写入 `Outputs` 下对应的 ASCII 作业目录。

## 环境

默认复用以下 Conda 环境：

```text
D:\Application\Anaconda\envs\sports3d
```

当前实现会使用该环境里的 `sports2d.exe`、PySide6、Plotly、pandas、openpyxl、OpenSim、Pose2Sim，以及系统 PATH 中的 `ffmpeg/ffprobe`。

## 使用方式

1. 在 `Inputs` 下为每次分析新建一个文件夹，并把视频放进去。

   ```text
   Inputs/
     20260620_Squat/
       squat_side.mp4
   ```

2. 双击运行：

   ```text
   run_gui.bat
   ```

3. 在左侧勾选要分析的输入作业。
4. 在右侧标签页填写参数。默认模式会开启标记增强和 OpenSim 逆运动学；如果只想快速检查，可手动关闭。
5. 点击“运行所选分析”。
6. 在 `Outputs\<ASCII作业名>` 查看结果。

## GUI 参数说明

- 基础信息：身高、体重、检测人数、可见侧、时间范围、慢动作倍率。
- 输出：处理后视频、逐帧图片、TRC、MOT、C3D、Sports2D 原生图表。
- 姿态检测：姿态模型、检测模式、跟踪模式、计算设备、推理后端、置信度阈值。
- 尺度/标定：像素到米、地面角、XY 原点、透视参数、标定文件。
- 后处理：插值、离群值处理、滤波方式和 Butterworth 参数。
- 逆运动学：IK、标记增强、脚贴地修正、简单模型、OpenSim setup 路径。
- TOML 预览：实时查看即将传给 Sports2D 的配置。

## 输出内容

每个作业目录包含：

- `run_config.toml`：本次 Sports2D 配置。
- `run.log`：自动化子进程日志。
- `environment_report.json`：环境检查结果。
- `video_metadata.json`：原始视频和临时分析视频的元数据。
- `<video>_Sports2D/`：Sports2D 原生输出，包括 TRC、MOT、处理后视频、图表等。
- `<video>_Sports2D/reports/*_interactive.html`：离线交互式 HTML 报告。
- `analysis_report.xlsx`：Excel 数据与统计汇总。

## 视频方向处理

工具会先用 `ffprobe` 检查视频方向元数据。若发现 `rotation`，会用 `ffmpeg` 生成已物理旋转且清除旋转元数据的临时 MP4，再交给 Sports2D 分析。这用于避免竖屏视频在 OpenCV/Sports2D/OpenSim 链路中被错误当作横屏处理。

## 命令行

列出输入作业：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli list
```

快速跑一个作业，不启用 IK 和标记增强：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli run --job OfficialDemo --quick --start 0 --end 1
```

检查环境：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.cli check-env
```

## 注意

- 默认不提交 `Inputs` 和 `Outputs` 中的真实数据到 GitHub。
- DeepSort 选项保留，但当前环境如果缺少 `deep_sort_realtime` 和 `torchreid`，GUI 会提示需要额外安装依赖。
- “一键更新”只更新 `sports2d` 和 `pose2sim`，不会自动升级 OpenSim。
- Sports2D 的 2D 结果质量高度依赖拍摄角度、遮挡情况和姿态估计质量；重要结果仍建议在 OpenSim 中复核。
