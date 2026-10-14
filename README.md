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

ROS 2 仿真启动与三车控制：[src/ros2_simulation](src/ros2_simulation/README.md)。
