# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
import std_msgs.msg as std_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import time
import geometry_msgs.msg as geometry_msgs


# C++と同じく、Node型を継承します。
class PathFollower(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('path_follower_node')
        
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
        
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Path, '/potential_astar_path', self.get_path, qos_profile) #set subscribe pcd topic name
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_wheel', self.get_odom, qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # タイマーを0.05秒（50ミリ秒）ごとに呼び出す
        self.timer = self.create_timer(0.05, self.robot_ctrl)
        
        
        # Publisherを作成
        self.cmd_vel_publisher = self.create_publisher(geometry_msgs.Twist, 'cmd_vel', qos_profile) #set publish pcd topic name
        
        #パラメータ init
        self.path_plan = np.array([[0],[0],[0]])
        
        #positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        #path follow
        self.target_dist = 1.0
        self.stop_flag = 0
        
        #pd init
        self.e_n = 0;
        self.e_n1 = 0;
        self.k_p = 0.6;
        self.k_d = 0.3;
        
        
    def get_path(self, msg):
        self.get_logger().info('Received path with %d waypoints' % len(msg.poses))
        path_x=[];path_y=[];path_z=[];  
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z
            path_x = np.append(path_x,x)
            path_y = np.append(path_y,y)
            path_z = np.append(path_z,z)
        
        self.path_plan = np.vstack((path_x, path_y, path_z))
        
    def robot_ctrl(self):
        self.get_logger().info('0.05秒ごとに車両制御を実行')
        
        path = self.path_plan;
        position_x=self.position_x; position_y=self.position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        #set target_rad
        path_x_diff = path[0,:] - position_x
        path_y_diff = path[1,:] - position_y
        path_diff = np.sqrt(path_x_diff**2 + path_y_diff**2)
        path_diff_target_dist = np.abs(path_diff - self.target_dist)
        path_target_ind_sort = np.argsort(path_diff_target_dist)[:4] #check 4point
        target_ind = np.max(path_target_ind_sort)
        target_point = path[:,target_ind]
        relative_point_x = target_point[0] - position_x
        relative_point_y = target_point[1] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, target_point[2]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        target_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        target_theta = target_rad * (180 / math.pi)
        #set speed
        if abs(target_theta) < 10:
            speed = 0.4
        elif abs(target_theta)  < 27:
            speed = 0.3
        elif abs(target_theta)  > 90:
            speed = 0.2
        else:
            speed = 0.2
        self.get_logger().info('speed = %f' % (speed))
        
        target_rad_pd = self.sensim0(target_rad)
        
        #make msg
        twist_msg = geometry_msgs.Twist()
        #check stop flag
        if self.stop_flag == 0:
            twist_msg.linear.x = speed #0.3  # 前進速度 (m/s)
            twist_msg.angular.z = target_rad_pd  # 角速度 (rad/s)
        else:
            twist_msg.linear.x = 0.0  # 前進速度 (m/s)
            twist_msg.angular.z = 0.0  # 角速度 (rad/s)
        self.cmd_vel_publisher.publish(twist_msg)
        self.get_logger().info('Publishing cmd_vel: linear.x = %f, angular.z = %f : %f deg' % (twist_msg.linear.x, twist_msg.angular.z, math.degrees(twist_msg.angular.z)))
        
        
        
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
        
    def sensim0(self, steering):
        self.e_n = steering
        steering = (self.k_p * self.e_n + self.k_d*(self.e_n - self.e_n1))
        self.e_n1 = self.e_n
        return steering


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
        
# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    path_follower = PathFollower()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(path_follower)
    # 明示的にノードの終了処理を行います。
    path_follower.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
