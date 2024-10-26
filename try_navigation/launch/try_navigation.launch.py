import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    rviz_config_dir = os.path.join(
        get_package_share_directory('try_navigation'),
        'config', 'config.rviz')
    assert os.path.exists(rviz_config_dir)
    #get livox data    
    livox_to_pointcloud2_launch_file = os.path.join(
        get_package_share_directory('livox_to_pointcloud2'),
        'launch',
        'livox_to_pointcloud2.launch.py'
    )
    #fast_lio launch    
    #fast_lio_launch_file = os.path.join(
    #    get_package_share_directory('fast_lio'),
    #    'launch',
    #    'mapping.launch.py'
    #)
    
    
    return LaunchDescription([
        #rviz2
        Node(package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            output='screen'
        ),
        #get livox data
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(livox_to_pointcloud2_launch_file)
        ),
        
        #fast_lio launch 
        #IncludeLaunchDescription(
        #    PythonLaunchDescriptionSource(fast_lio_launch_file)
        #),
        #pcd rotation
        Node(package='pcd_convert',
            executable='pcd_rotation',
            name='pcd_rotation_node',
            output='screen',
            arguments=[]
        ),
        #fast_odom convert
        Node(package='fast_odom_convert',
            executable='fast_odom_convert',
            name='fast_odom_convert_node',
            output='screen',
            arguments=[]
        ),
        #pcd segmentation
        Node(package='pcd_convert',
            executable='pcd_height_segmentation',
            name='pcd_heigth_segmentation_node',
            output='screen',
            arguments=[]
        ),
        #odom wheel
        Node(package='try_navigation',
            executable='odom_wheel',
            name='odom_wheel_node',
            output='screen',
            arguments=[]
        ),
        
        ##waypoint manager
        Node(package='try_navigation',
            executable='waypoint_manager_maprun',
            name='waypoint_manager_maprun_node',
            output='screen',
            arguments=[],
        ),
        
        #reflection intensity map
        Node(package='try_navigation',
            executable='reflection_intensity_map',
            name='reflection_intensity_map_node',
            output='screen',
            arguments=[],
        ),
        #path planning
        Node(package='try_navigation',
            executable='potential_astar',
            name='potential_astar_node',
            output='screen',
            arguments=[],
        ),
        #robot ctrl
        Node(package='try_navigation',
            executable='path_follower',
            name='path_follower_node',
            output='screen',
            arguments=[],
        ),
    ])
