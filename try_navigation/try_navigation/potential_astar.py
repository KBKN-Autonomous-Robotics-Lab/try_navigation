# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
# ROS 2の文字列型を使えるようにimport
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
from livox_ros_driver2.msg import CustomMsg
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd
#import open3d as o3d
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
import cv2
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import time
import matplotlib.pyplot
import struct
import geometry_msgs.msg as geometry_msgs

from scipy import interpolate
from std_msgs.msg import Float32MultiArray
import cv2



#map save
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global -f ~/ros2_ws/src/map/test_map --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global --occ 0.10 --free 0.05 -f ~/ros2_ws/src/map/test_map2 --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#--occ:  occupied_thresh  この閾値よりも大きい占有確率を持つピクセルは、完全に占有されていると見なされます。
#--free: free_thresh	  占有確率がこの閾値未満のピクセルは、完全に占有されていないと見なされます。

# C++と同じく、Node型を継承します。
class PotentialAStar(Node):
    # コンストラクタです、Mid360Subscriberクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。（https://www.python-izm.com/advanced/class_extend/）今回の場合継承するクラスはNodeになります。
        super().__init__('potential_astar_node')
        
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        
        # Subscriptionを作成。CustomMsg型,'/livox/lidar'という名前のtopicをsubscribe。
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_obs', self.potential_astar, qos_profile)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_wheel', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(geometry_msgs.PoseArray,'/current_waypoint', self.get_waypoint, qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        # Publisherを作成
        self.potential_astar_path_publisher = self.create_publisher(nav_msgs.Path, 'potential_astar_path', qos_profile)
        self.pcd_obs_global_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_obs_global', qos_profile) 
        #パラメータ
        #mid360 positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        #mid360 buff
        self.pcd_ground_buff = np.array([[],[],[],[]]);
        
        #potential astar
        #241025パラメータ:self.cg=20; self.lg=20; self.co=20; self.lo=0.55;
        self.cg=20 #ポテンシャルの引力パラメータ
        self.lg=20 #ポテンシャルの引力パラメータ
        self.co=11 #ポテンシャルの斥力パラメータ SICKパラ目：co=11;lo=0.55;
        self.lo=0.55 #0.5#0.9#ポテンシャルの斥力パラメータ
        self.est_xy = [0,0]#自己位置仮入力
        self.wp_xy = [10,0]#ウェイポイント仮入力
        self.astar_path = [0,10]#ウェイポイント仮入力
        self.obs_pixel = 100/5#障害物のグリッドサイズ設定
        self.obs_range = 10#障害物情報の範囲
        self.obs_judge = 0#obs_judgeより大きい場合障害物ありと判定する
        
        #waypoint
        self.waypoint_xy = np.array([[10],[0],[0]])
        
    def timer_callback(self):
        #
        pass
        
    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.theta_x = 0 #roll /math.pi*180
        self.theta_y = 0 #pitch /math.pi*180
        self.theta_z = yaw /math.pi*180
        
	#self.est_xy = np.array([self.position_x, self.position_y])
	#self.est_xy = np.array([0, 0]) #test param
	
    def get_waypoint(self, msg):
        #get waypoint
        self.waypoint = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses]).T
        
        #relative waypoint
        relative_point_x = self.waypoint[0] - self.position_x
        relative_point_y = self.waypoint[1] - self.position_y
        relative_point = np.array((relative_point_x, relative_point_y, self.waypoint[2]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, self.theta_x, self.theta_y, -self.theta_z)
        
        self.wp_xy = [relative_point_rot[0], relative_point_rot[1]]
	
    def pointcloud2_to_array(self, cloud_msg):
        # Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        
        return point_cloud_matrix
        
    def potential_astar(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        #obs round&duplicated  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))
        points_round = np.round(obs_points * self.obs_pixel) / self.obs_pixel
        obs_xy_local = points_round[:,~pd.DataFrame({"x":points_round[0,:], "y":points_round[1,:]}).duplicated()]
        obs_xy = np.vstack((obs_xy_local[0,:], obs_xy_local[1,:]))
        
        reflect_set = points[3,~pd.DataFrame({"x":points_round[0,:], "y":points_round[1,:]}).duplicated()]
        #obs global
        obs_xy_rot, obs_rot_matrix = rotation_xyz(obs_xy_local, self.theta_x, self.theta_y, self.theta_z)
        obs_x_grobal = obs_xy_rot[0,:] + self.position_x
        obs_y_grobal = obs_xy_rot[1,:] + self.position_y
        obs_global = np.vstack((obs_x_grobal, obs_y_grobal, obs_xy_local[2,:], reflect_set) , dtype=np.float32)
        print(f"obs_xy ={obs_xy.shape}")
        print(f"obs_global ={obs_global.shape}")
        print(f"obs_global ={obs_global.dtype}")
        #set self position
        #self.est_xy = [self.position_x, self.position_y]
        print(f"self.est_xy ={self.est_xy}")
        
        astar_path = self.path_plan(obs_xy)
        print(f"astar_path ={astar_path.shape}")
        astar_path = np.vstack((astar_path, np.zeros([1,len(astar_path[0,:])]) ))
        print(f"astar_path ={astar_path.shape}")
        astar_path_rot, astar_path_rot_matrix = rotation_xyz(astar_path, self.theta_x, self.theta_y, self.theta_z)
        astar_path_x_grobal = astar_path_rot[0,:] + self.position_x
        astar_path_y_grobal = astar_path_rot[1,:] + self.position_y
        astar_path_grobal = np.vstack((astar_path_x_grobal, astar_path_y_grobal))
        
        #publish for rviz2
        #global map rviz2
        potential_astar_path = path_msg(astar_path_grobal, t_stamp, 'odom')
        self.potential_astar_path_publisher.publish(potential_astar_path)    
        #global obs rviz2
        obs_global_msg = point_cloud_intensity_msg(obs_global.T, t_stamp, 'odom')
        self.pcd_obs_global_publisher.publish(obs_global_msg) 
		
    def path_plan(self, obs_xy):
        #process: 検索マップ準備
        astar_x =  np.arange(-9.9, 9.9,  0.3) +self.est_xy[0]	#%%Astarのxを定義
        astar_y =  np.arange(9.9, -9.9, -0.3) +self.est_xy[1]	#%%Astarのyを定義
        astar_xy = np.ones([len(astar_x),len(astar_y)])*500     #%%マップを用意 500は暫定？コスト計算のReturnで使ってる？
        astar_xy_find = np.zeros([len(astar_y),len(astar_x)])	#%%一度通過した箇所を記憶
        astar_xn = round(len(astar_x)/2)		#%%探索を行うx座標
        astar_yn = round(len(astar_y)/2)		#%%探索を行うy座標
        astar_xy_find[astar_yn, astar_xn] = 1		#%%初期座標の通過設定
        astar_count = 0							#%%ループ回数チェック用
	
	
        #■  process : A-star search
        astar_path_x=np.array([astar_xn])
        astar_path_y=np.array([astar_yn])
        #現在地より9m先の経路まで生成
        while (astar_xy[astar_xn, astar_yn]>0.1) and ((abs(astar_x[astar_xn]) - self.est_xy[0]) < 9) and ((abs(astar_y[astar_yn]) - self.est_xy[1]) < 9):
            astar_x_search = np.array([astar_x[astar_xn], astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1], astar_x[astar_xn]]) # 探索するx座標[[0, 1, 0],[2, 3, 4],[0, 5, 0]]：0は見ない 1-5の順に十字検索
            astar_y_search = np.array([astar_y[astar_yn-1], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn+1]]) # 探索するy座標[[0, 1, 0],[2, 3, 4],[0, 5, 0]]：0は見ない 1-5の順に十字検索
            astar_ug=self.cg*(1-np.exp(-( (astar_x_search-self.wp_xy[0])**2+(astar_y_search-self.wp_xy[1])**2 )/self.lg**2)) #引力計算
            obs_short = ( np.abs(obs_xy[0]-astar_x[astar_xn]) + np.abs(obs_xy[1]-astar_y[astar_yn]) ) < 3 # 2　前は2だったけどとりあえずテスト中は5に
            obs_short_x = obs_xy[0, np.array(obs_short) ]
            obs_short_y = obs_xy[1, np.array(obs_short) ]
            astar_uo_x = astar_x_search -  obs_xy[0].reshape(len(obs_xy[0]),1) #x-xo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
            astar_uo_y = astar_y_search -  obs_xy[1].reshape(len(obs_xy[1]),1) #y-yo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
            astar_uo_x2 = ( astar_uo_x * np.ones([len(obs_xy[0]),len(astar_x_search)]) ) ** 2 #(x-xo)^2
            astar_uo_y2 = ( astar_uo_y * np.ones([len(obs_xy[1]),len(astar_y_search)]) ) ** 2 #(y-yo)^2
            astar_uo = sum(self.co * np.exp(- (astar_uo_x2 + astar_uo_y2) / self.lo**2 ) ) # Uo計算 sum{co*e(-((x-xo)^2+(y-yo)^2)/lo^2)}
            astar_u=(1/self.cg*astar_uo+1)*astar_ug #UgとUoでポテンシャル計算
            astar_xy[[astar_xn,astar_xn-1,astar_xn,astar_xn+1,astar_xn],[astar_yn-1,astar_yn,astar_yn,astar_yn,astar_yn+1]] = astar_u #代入
            astar_xymin = astar_xy + (astar_xy_find*500) #一度通過した点を除外
            astar_xymin_ind = np.unravel_index(np.argmin(astar_xymin), astar_xymin.shape) #最もポテンシャルの低い場所のIndexを探す
            astar_xn = astar_xymin_ind[0] #次に探索を行うx座標を指定
            astar_yn = astar_xymin_ind[1] #次に探索を行うy座標を指定
            astar_xy_find[astar_xn, astar_yn] = 1 #探索座標の通過設定
            astar_path_x = np.append(astar_path_x, astar_xn) #x座標の通過記録
            astar_path_y = np.append(astar_path_y, astar_yn) #y座標の通過記録
            
            astar_count = astar_count + 1 #ループカウント
            if astar_xy[astar_yn, astar_xn] <0.2:
                self.get_logger().info(f"Goal: astar_path_x, astar_path_y ={astar_path_x, astar_path_y}")
                break
            if astar_count > 100:
                self.get_logger().info("Count Break")
                break
                
        #■  process : A-star Return
        astar_xy_rev = np.ones([len(astar_y),len(astar_x)])*500		#%%リターンのマップを用意
        astar_xy_find_rev = np.zeros([len(astar_y),len(astar_x)])		#%%リターンの一度通過した箇所を記憶
        astar_xy_find_rev[astar_xn, astar_yn]  = 1		#%%初期座標の通過設定
        astar_path_x_rev = np.array([astar_xn])
        astar_path_y_rev = np.array([astar_yn])
        while not( (astar_xn == round(len(astar_x)/2) ) and (astar_yn == round(len(astar_y)/2)) ):
            astar_x_search_rev = np.array([astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1], astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1], astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1] ]) # 探索するx座標[[1, 2, 3],[4, 5, 6],[7, 8, 9]] 1-9の順に十字検索
            astar_y_search_rev = np.array([astar_y[astar_yn-1],astar_y[astar_yn-1], astar_y[astar_yn-1], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn+1], astar_y[astar_yn+1], astar_y[astar_yn+1]]) # 探索するy座標[[1, 2, 3],[4, 5, 6],[7, 8, 9]]： 1-9の順に十字検索
            astar_xy_rev[ [astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1 ], [ astar_yn-1, astar_yn-1, astar_yn-1, astar_yn, astar_yn, astar_yn, astar_yn+1, astar_yn+1, astar_yn+1 ] ] = np.sqrt( ( (astar_x_search_rev - self.est_xy[0]) ** 2 ) + ( (astar_y_search_rev - self.est_xy[1]) ** 2 ) ) +500 - 500*astar_xy_find[ [astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1 ], [ astar_yn-1, astar_yn-1, astar_yn-1, astar_yn, astar_yn, astar_yn, astar_yn+1, astar_yn+1, astar_yn+1 ] ]#自己位置から探索点までの距離代入
            astar_xymin_rev = ( (astar_xy_rev + 500 * astar_xy_find_rev) )# -500 * astar_xy_find		#一度通過した箇所を除外
            astar_xymin_rev_ind = np.unravel_index(np.argmin(astar_xymin_rev), astar_xymin_rev.shape) #最も開始点に近い場所のIndexを探す
            astar_xn = astar_xymin_rev_ind[0] #次に探索を行うx座標を指定
            astar_yn = astar_xymin_rev_ind[1] #次に探索を行うy座標を指定
            astar_xy_find_rev[astar_xn, astar_yn]  = 1		#%%初期座標の通過設定
            astar_path_x_rev = np.append(astar_path_x_rev, astar_xn) #x座標の通過記録
            astar_path_y_rev = np.append(astar_path_y_rev, astar_yn) #y座標の通過記録
                

        #■  process : A-star path planning 
        astar_path_x_rev2 = astar_path_x_rev[::-1][1:len(astar_path_x_rev)-1] #indexが逆に入っているので元に戻す 自己位置とWaypoint位置は除外
        astar_path_y_rev2 = astar_path_y_rev[::-1][1:len(astar_path_y_rev)-1] #indexが逆に入っているので元に戻す 自己位置とWaypoint位置は除外
        astar_judge_x = astar_x[astar_path_x_rev2] -  obs_xy[0].reshape(len(obs_xy[0]),1) #x-xo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
        astar_judge_y = astar_y[astar_path_y_rev2] -  obs_xy[1].reshape(len(obs_xy[1]),1) #y-yo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
        astar_judge_x2 = ( astar_judge_x * np.ones([len(obs_xy[0]),len(astar_x[astar_path_x_rev2])]) ) ** 2 #(x-xo)^2
        astar_judge_y2 = ( astar_judge_y * np.ones([len(obs_xy[1]),len(astar_y[astar_path_y_rev2])]) ) ** 2 #(y-yo)^2
        self.get_logger().info(f"astar_judge_x2 ={len(astar_judge_x2)}")
        self.get_logger().info(f"astar_path_x_rev2 ={len(astar_path_x_rev2)}")
        if len(astar_judge_x2) > 0:
            astar_judge_obs_ind = np.minimum.reduce( np.sqrt(astar_judge_x2 + astar_judge_y2) ) <1.8#1.6
            self.get_logger().info(f"astar_judge_obs_ind ={astar_judge_obs_ind}")
            astar_path_point_x = np.append(np.append(self.est_xy[0],astar_x[astar_path_x_rev2[astar_judge_obs_ind]]),self.wp_xy[0])
            astar_path_point_y = np.append(np.append(self.est_xy[1],astar_y[astar_path_y_rev2[astar_judge_obs_ind]]),self.wp_xy[1])
        else:
            astar_path_point_x = np.append(np.append(self.est_xy[0],astar_x[astar_path_x_rev2[:]]),self.wp_xy[0])
            astar_path_point_y = np.append(np.append(self.est_xy[1],astar_y[astar_path_y_rev2[:]]),self.wp_xy[1])
            
        astar_dist = np.append(0, np.cumsum( np.sqrt(np.diff(astar_path_point_x)**2 + np.diff(astar_path_point_y)**2) ) ) #x/y軸を距離軸で2次元表現 前後のポイント間距離を計算

        astar_interp_x = interpolate.interp1d(astar_dist, astar_path_point_x, kind='linear')
        astar_interp_y = interpolate.interp1d(astar_dist, astar_path_point_y, kind='linear')
        astar_interp_list = np.linspace(0,astar_dist[len(astar_dist)-1],round(astar_dist[len(astar_dist)-1]/0.5) )
        astar_path_x = astar_interp_x(astar_interp_list)
        astar_path_y = astar_interp_y(astar_interp_list)
        astar_path = np.append(np.append(astar_path_x,self.wp_xy[0]),np.append(astar_path_y,self.wp_xy[1])).reshape(2,len(astar_path_x)+1)
                
        return astar_path     
        
def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])
    
    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])
    
    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    #print(f"rot_matrix ={rot_matrix}")
    #print(f"pointcloud ={pointcloud.shape}")
    rot_pointcloud = rot_matrix.dot(pointcloud)
    return rot_pointcloud, rot_matrix

def quaternion_to_euler(x, y, z, w):
    # クォータニオンから回転行列を計算
    rot_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])

    # 回転行列からオイラー角を抽出
    roll = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
    pitch = np.arctan2(-rot_matrix[2, 0], np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2))
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return roll, pitch, yaw
    

def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.
    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [
            sensor_msgs.PointField(name='x', offset=0, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='y', offset=4, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='z', offset=8, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
        ]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = std_msgs.Header(frame_id=parent_frame, stamp=t_stamp)
    

    return sensor_msgs.PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4), # Every point consists of three float32s.
        row_step=(itemsize * 4 * points.shape[0]), 
        data=data
    )

def path_msg(waypoints, stamp, parent_frame):
    wp_msg = nav_msgs.Path()
    wp_msg.header.frame_id = parent_frame
    wp_msg.header.stamp = stamp
        
    # ウェイポイントを追加
    for i in range(waypoints.shape[1]):
        waypoint = geometry_msgs.PoseStamped()
        waypoint.header.frame_id = parent_frame
        waypoint.header.stamp = stamp
        waypoint.pose.position.x = waypoints[0, i]
        waypoint.pose.position.y = waypoints[1, i]
        waypoint.pose.position.z = 0.0
        waypoint.pose.orientation.w = 1.0
        wp_msg.poses.append(waypoint)
    return wp_msg


# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    potential_astar = PotentialAStar()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(potential_astar)
    # 明示的にノードの終了処理を行います。
    potential_astar.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
