import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,IncludeLaunchDescription,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def prepare(context):
    from pathlib import Path
    roots=os.environ.get('GAZEBO_MODEL_PATH','').split(':')
    if not any((Path(p)/'rmuc2026_v20_native/model.sdf').is_file() for p in roots):
        raise FileNotFoundError('Verified RMUC2026 V2.0.0 model missing; use run.sh')
    return []

def generate_launch_description():
    share=get_package_share_directory('rmuc26_assessment')
    return LaunchDescription([
        DeclareLaunchArgument('gui',default_value='true'),DeclareLaunchArgument('rviz',default_value='true'),
        OpaqueFunction(function=prepare),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('gazebo_ros'),'launch/gazebo.launch.py')),launch_arguments={'world':os.path.join(share,'worlds/rmuc26.world'),'gui':LaunchConfiguration('gui'),'verbose':'true'}.items()),
        Node(package='rmuc26_assessment',executable='view',parameters=[{'use_sim_time':True,'model_path':os.environ.get('RMUC_MODEL_DIR','')}]),
        Node(package='rviz2',executable='rviz2',arguments=['-d',os.path.join(share,'config/assessment.rviz')],parameters=[{'use_sim_time':True}],condition=IfCondition(LaunchConfiguration('rviz'))),
        Node(package='tf2_ros',executable='static_transform_publisher',name='lidar_tf',arguments=['--x','0','--y','0.4','--z','0.1008503151','--frame-id','world','--child-frame-id','astra_lidar']),
    ])
