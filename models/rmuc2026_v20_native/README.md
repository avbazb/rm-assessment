# RMUC 2026 场地模型来源

本目录仅保存模型来源和校验信息，不附带官方 CAD、转换后的网格或场地模型本体。

来源：[RoboMaster 官方 RMUC 2026 场地 V2.0.0 CAD](https://bbs-web-static.robomaster.com/f637e13eadfc4602a76c7a05c54f71271782803452461/RMUC2026_V2.0.0.stp)。现有原生 CAD 转换模型保留 446 个部件及 428 个网格，`manifest.json` 记录源 CAD 和 430 个模型文件的大小、SHA-256 及相对路径。

准备与清单一致的 `rmuc2026_v20_native` Gazebo 模型目录，目录中需包含 `model.sdf`、`model.config` 和完整的 `meshes`。在 ROS 2 环境中运行：

```sh
RMUC_MODEL_DIR=/path/to/rmuc2026_v20_native bash src/ros2_simulation/run.sh launch
```

场地 CAD 及派生模型归 DJI / RoboMaster 原权利人所有，获取与使用遵守官方条款。本仓库不重新授权这些资产。
