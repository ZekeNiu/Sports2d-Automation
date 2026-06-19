# Sports2D 自动化中文桌面工具

这个项目为 Sports2D 提供中文桌面 GUI、批处理流水线、HTML 交互报告和 Excel 汇总报告。默认环境为：

```text
D:\Application\Anaconda\envs\sports3d
```

启动方式：

```powershell
run_gui.bat
```

或：

```powershell
D:\Application\Anaconda\envs\sports3d\python.exe -m sports2d_automation.gui
```

## 基本流程

1. 在 `Inputs` 下为每个分析任务新建一个文件夹，把视频放进去。
2. 启动 GUI，左侧勾选一个或多个输入作业。
3. 选择分析预设：
   - `推荐新手模式`：默认模式，生成 2D 角度、处理后视频、TRC/MOT、HTML 和 Excel，不默认运行 OpenSim IK。
   - `完整 OpenSim 模式`：额外运行 OpenSim IK，并默认启用标记增强。该模式仍需查看质量诊断。
   - `专家模式`：显示底层参数，允许调整推理后端、置信度阈值、标定、滤波和 OpenSim setup 等高风险参数。
4. 填写身高、体重、可见侧、时间范围等必要信息。
5. 点击“运行所选分析”。

## 结果解释原则

- 本工具当前最可靠的核心产出是 HTML/Excel 中的 Sports2D 原生 2D 视频平面角。
- 处理后视频骨架稳定，说明 2D pose 检测稳定；它不能单独证明 OpenSim IK 或三维 MOT 动作准确。
- Sports2D 原生角度不是完整三维屈曲、外展、旋转角。侧面拍摄时，它可用于描述拍摄平面内的屈伸趋势；正面、背面或斜拍时，应解释为视频平面角。
- OpenSim `*_ik.mot` 若存在，会作为附加统计展示。只有 marker error、尺度、拍摄方向和标定均合理时，才建议进一步用于三维解释。
- HTML 报告不再展示三维骨架模型。单目二维视频推断出的三维骨架容易造成误读，因此报告改为重点展示可核查的 2D 角度、质量诊断和动作含义说明。

## GUI 参数策略

默认界面只展示普通用户需要理解和负责的参数。以下内容由系统自动设置或隐藏到专家模式：

- 普通视频宽高：自动从视频元数据读取，不要求手动输入。
- 推理后端、计算设备、跟踪模式、置信度阈值：默认自动或 Sports2D 推荐值。
- 地面角、XY 原点、透视参数、标定文件：除非已有可靠标定或报告提示地面估计异常，否则不建议修改。
- 插值、离群值、滤波、片段保留：默认使用保守后处理；只在专家模式中开放。
- OpenSim setup、极值裁剪、临时文件清理：只在专家模式中开放。

每个可见参数旁都有 `?` 说明按钮，包含用途、默认建议、选项说明、何时修改和设置不当的风险。

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

## HTML 报告内容

- 视频核对：播放处理后视频，用于确认 2D 骨架是否跟随正确受试者。
- 角度曲线：Plotly 离线交互曲线，鼠标悬停可同步视频时间。
- 重点关节指标：用户可勾选关注的关节或节段，卡片显示最小值、最大值、ROM、均值和峰值时间。
- 质量诊断：显示可见侧、camera horizon、OpenSim marker error 等提示。
- 角度定义与动作含义：解释每一列角度到底代表什么运动，例如膝屈曲/伸展趋势、髋屈曲/伸展趋势、节段相对水平线角度等。
- 完整统计表：列出所有角度的最小值、最大值、均值、标准差、ROM 和峰值时间。

## Excel 报告内容

- 每个 MOT 文件的原始数据表。
- 角度统计表：最小值、最大值、均值、标准差、ROM、峰值时间、动作含义和解释文本。
- 质量诊断、参数、环境、视频元数据和输出文件索引。

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
