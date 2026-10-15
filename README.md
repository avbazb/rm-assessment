本仓库用于rm考核

## 项目结构

```text
.
├── src/
│   ├── preview/          # 检测预览、传统视觉、融合、跟踪与预测代码
│   ├── assessment/       # 反小陀螺、吊射等考核算法代码
│   └── ros2_simulation/  # RMUC26地图、三车独立控制、Gazebo与RViz一键启动
├── models/
│   ├── recommended.onnx  # 运行模型
│   ├── recommended.pt    # 原始权重
│   ├── recommended.json  # 导出参数
│   ├── manifest.json     # 视觉模型文件与校验信息
│   └── rmuc2026_v20_native/ # 仿真地图来源与导入校验信息
├── run.py                # 预览程序入口
└── requirements.txt      # Python依赖
```

## 运行视觉预览

使用 64 位 Python 3.12 或更高版本。在仓库目录打开 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

启动后自动打开本地网页，可上传图片或视频，查看检测、运动预测、三维位姿和落点偏差。源码仓库不附带演示素材，首次启动素材库为空。

图片检测仅需上述 Python 依赖。视频上传还需要安装 FFmpeg，并确保 `ffmpeg` 和 `ffprobe` 都在 PATH 中。

检测预览与运动预测页面支持选择 0–500 毫秒的提前量，显示预测位置、目标 ID、位置、速度和历史轨迹。视频播放按解码帧刷新，跟踪状态外推到当前画面时间；暂停后显示对应检测帧的原始框和灯条。

检测接口 `/api/analyze` 返回 `motion_estimates` 和 `future_prediction`。建立跟踪段后，可通过 `/api/prediction?session=demo&epoch=1&horizon=0.5` 单独读取预测位置；`session` 和 `epoch` 与检测请求一致，单位为图像像素。`usable` 表示是否通过当前发布条件；`estimated` 保留估计坐标及不确定程度，超出画面或观测过期时不显示预测点。

运行自检：`.\.venv\Scripts\python.exe run.py --self-test`。没有内置素材时使用空白图检查运行链路；也可用 `--self-test-image '图片路径'` 检查真实图片。

## ROS 2 仿真

仿真环境、地图准备、一键启动与三车控制见 [src/ros2_simulation](src/ros2_simulation/README.md)。
