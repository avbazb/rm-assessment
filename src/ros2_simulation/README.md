# ROS 2 仿真考核

RMUC26 地图、三台独立差速小车、Gazebo Classic 11 和 RViz2 共用一个 launch。三车初始位置分别为 `(4,-0.7)`、`(5,0.4)`、`(4,1.5)` 米。

## 依赖

Ubuntu 22.04 + ROS 2 Humble（Windows 使用 WSL2/WSLg）。已安装 ROS 2 软件源后：

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-rviz2 ros-humble-tf2-ros \
  python3-colcon-common-extensions python3-numpy python3-opencv
```

地图使用原生 `rmuc2026_v20_native` Gazebo 模型目录，内含 `model.sdf`、`model.config` 和 `meshes`。来源及全部430文件的校验信息在 `models/rmuc2026_v20_native`；官方 CAD 及其派生网格不随代码重新分发。

## 一键启动

在本目录打开 PowerShell，第一次指定已有地图模型目录：

```powershell
.\run.ps1 -ModelDir '你的模型目录\rmuc2026_v20_native'
```

通过校验后自动记住地图目录。以后双击 `run.bat` 即可同时启动 Gazebo 和 RViz2，第一次自动编译。

Linux：

```bash
RMUC_MODEL_DIR=/path/to/rmuc2026_v20_native bash run.sh launch
```

无界面运行：`run.ps1 -Headless` 或 `bash run.sh launch gui:=false rviz:=false`。重新编译：`run.ps1 -Action build`。端口及工作空间可用 `GAZEBO_MASTER_URI`、`ROS_DOMAIN_ID` 和 `RMUC_WORKSPACE` 环境变量设置。

Windows 启动脚本会将这些设置传入 WSL；`RMUC_WORKSPACE` 使用 Linux 路径。更换仓库目录时，可设置一个新的工作空间，避免复用另一个目录的编译缓存：

```powershell
$env:RMUC_WORKSPACE='/tmp/rm-assessment-workspace'
.\run.ps1 -ModelDir '你的模型目录\rmuc2026_v20_native'
```

## 三车独立控制

另开 ROS 2 终端：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=41 ROS_LOCALHOST_ONLY=1
ros2 topic pub --rate 10 /assessment/car1/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.12}, angular: {z: 0.0}}'
```

把 `car1` 改为 `car2` 或 `car3`。停止发布后发送零速度：

```bash
ros2 topic pub --once /assessment/car1/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.0}}'
```

## 运行检查

保持仿真运行，在本目录另开 PowerShell：

```powershell
.\run.ps1 -Action verify
.\run.ps1 -Action verify-display
```

检查逐车移动、其他车辆漂移、同时移动、转向、停止、初始间距、状态连续性、相机和雷达消息，以及 RViz 订阅。结果保存在本地 `evidence/`，不提交到仓库。控制只发送速度指令，不写模型位姿。

相机为60°水平视场、640×480、RGB、0.1–100米裁剪。二维单线雷达为70.4°水平视场、50×1采样、0.05–90米、1厘米距离分辨率、10Hz。
