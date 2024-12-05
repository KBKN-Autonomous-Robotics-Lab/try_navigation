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
from rclpy.action import ActionServer ####
from my_msgs.action import StopFlag ####


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
            depth = 1
        ) # BEST_EFFORT
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )

        # actionサーバーの生成(tuika)
        self.server = ActionServer(self,
            StopFlag, "stop_flag", self.listener_callback)
        
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Path, '/potential_astar_path', self.get_path, qos_profile) #set subscribe pcd topic name
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_wheel', self.get_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/fusion/odom', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_ekf_match', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_ref_slam', self.get_odom_ref, qos_profile_sub)
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_obs', self.obs_steer, qos_profile)
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
        
        #positon init
        self.ref_position_x = 0.0 #[m]
        self.ref_position_y = 0.0 #[m]
        self.ref_position_z = 0.0 #[m]
        self.ref_theta_x = 0.0 #[deg]
        self.ref_theta_y = 0.0 #[deg]
        self.ref_theta_z = 0.0 #[deg]
        
        #path follow
        self.target_dist = 1.2
        self.target_dist_near = 0.4
        self.stop_flag = 1
        
        #pd init
        self.e_n = 0;
        self.e_n1 = 0;
        self.k_p = 0.6;
        self.k_d = 0.3;
        
        self.stop_xy_test = [8, 10, -10, 10]
        self.stop_xy_test_flag = 1
        
        self.stop_xy = np.array([ #xmin,xmax,ymin,ymax
            #[  32,  37.1,    -7,     3, 1.0], #nakaniwa_1
            #[  38.7,  49,    -10, 10], #nakaniwa_1129_last
            #[ 44.2, 64.2, 27.7, 32.7, 1.0], #nakaniwa_2
            #[  2.9,  7.9,  42.8,  62.8], #nakaniwa_3
            
            #[   4.8,   8.9,  53.1,  57.1], #nakaniwa_4
            #[   8,   13, -10,  10], #test
            #[  15,   19, -10,  10], #test
            #[ -25,  -20, -30, -25], #shiyakusyo
            #[-235, -225,  43,  47], #hodu1 temae 
            #[-235, -225,  57,  61], #hodou1
            #[-235, -225,  75,  79], #hodou2 temae
            #[-235, -233,  70,  80], #hodou2
            #[-256, -254,  70,  80], #hodou3 temae
            #[-250, -248,  70,  80], #hodou3
            #[-240, -230,  71,  73], #hodou4 temae
            #[-240, -230,  68,  70], #hodou4
            #[ -55,  -35,  40,  42], #Goal
            
            #xmin,      xmax,    ymin,   ymax, flag
            [-28.8,	-22.8,	-33.3,	-23.3, 1.0], #shiyakusyo ||| before:[-28.8,	-23.8,	-33.3,	-23.3, 1.0], 
            [-64,	-59,	-47.5,	-27.5, 1.0], #tyokusen1
            [-141.5,	-136.5,	-49,	-29,   1.0], #tyokusen2
            [-240,	-220,	51.9,	56.9,  1.0], #singou1 temae
            [-240,	-220,	56.5,	61.5,  1.0], #singou1
            [-240,	-220,	76.5,	82.3,  1.0], #singou2 temae ||| before:[-240,	-220,	77.3,	82.3,  1.0], 
            [-235,	-233,	69,	89,    1.0], #singou2
            [-300,	-240,	110,	140,   0.0], #through park
            [-253.0,	-250,	66,	92,    1.0], #singou3 temae ||| before:[-252.0,	-250,	69,	89,    1.0], 
            [-250.0,	-244,	66,	92,    1.0], #singou3       ||| before:[-249.0,	-244,	69,	89,    1.0], 
            [-240,	-220,	72,	74,    1.0], #singou4 temae
            [-240,	-220,	68.4,	70.4,  1.0], #singou4
            [-145.5,	-140,	-47.5,	-27.5, 1.0], #tyokusen3 ||| before:[-145.0,	-140,	-47.5,	-27.5, 1.0],
            [-67.5,	-62,	-47.5,	-27.5, 1.0], #tyokusen4 ||| before:[-67.0,	-62,	-47.5,	-27.5, 1.0], 
            [-55,	-35,	41,	46,    1.0], #Goal
            
            
            [ 999,  999, 999, 999, 0.0] ]) #
        self.stop_num = 0;
        
        #obs
        self.obs_points = np.array([[],[],[],[]])
        self.rh_obs = 0
        self.lh_obs = 0
            
        
    # actionリクエストの受信時に呼ばれる(tuika)
    def listener_callback(self, goal_handle):
        self.get_logger().info(f"Received goal with a: {goal_handle.request.a}, b: {goal_handle.request.b}")
        
        # クライアントから送られたaをstop_flagに代入
        self.stop_flag = goal_handle.request.a
        print(f"stop_flag set to: {self.stop_flag}")
        
        
        # フィードバックの返信
        for i in range(1):
            feedback = StopFlag.Feedback()
            feedback.rate = i * 0.1
            goal_handle.publish_feedback(feedback)
            time.sleep(0.5)

        # レスポンスの返信
        goal_handle.succeed()
        result = StopFlag.Result()
        result.sum = goal_handle.request.a + goal_handle.request.b  # 結果の計算
        return result         
        
    def get_path(self, msg):
        #self.get_logger().info('Received path with %d waypoints' % len(msg.poses))
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
        #self.get_logger().info('0.05秒ごとに車両制御を実行')
        
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
        target_theta = (target_rad) * (180 / math.pi)
        
        
        
        #set speed
        speed = 0.55
        lim_steer = 20
        #lim_steer = 30 24/11/29 ok
        if abs(target_theta) < 10:
            speed = 0.55
        elif target_theta  < -lim_steer:
            speed = 0.15
            target_rad = -lim_steer/180*math.pi
        elif lim_steer < target_theta:
            speed = 0.15
            target_rad = lim_steer/180*math.pi
        
        if abs(target_theta)  > 90:
            speed = -0.15
        
        
        
        points = self.obs_points 
        obs_theta = np.arctan2(points[1,:],points[0,:]) * 180/math.pi #arctan2(y,x)
        obs_dist = np.sqrt(points[0,:]**2 + points[1,:]**2)
        
        r_obs = (-120<obs_theta) * (obs_theta< -60) * (obs_dist<0.9)
        l_obs = (  60<obs_theta) * (obs_theta< 120) * (obs_dist<0.9)
        c_obs = ( -60<obs_theta) * (obs_theta<  60) * (obs_dist<1.2)
        c_obs_near = ( -50<obs_theta) * (obs_theta<  50) * (obs_dist<0.5)
        
        if np.any(r_obs) and np.any(l_obs) and ~np.any(c_obs) :
            speed = 0.3
            target_theta = 0
            target_rad = target_theta/180*math.pi
            #self.get_logger().info('||||||||||| center |||||||||||||||||')
        elif np.any(r_obs) and ~np.any(c_obs) :
            speed = 0.15
            if target_theta < 0:
                target_theta = 0.0#lim_steer
                target_rad = target_theta/180*math.pi
            #self.get_logger().info('lllllllllll go left lllllllllllllllll')
        elif np.any(l_obs) and ~np.any(c_obs) :
            speed = 0.15
            if 0 < target_theta:
                target_theta = 0.0#-lim_steer
                target_rad = target_theta/180*math.pi
            #self.get_logger().info('rrrrrrrrrr go right rrrrrrrrrrrrrrrrr')
        elif np.any(r_obs) and ~np.any(l_obs) and np.any(c_obs_near) :
            speed = 0.0
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            if abs(target_theta) < 3:
                speed = -0.15
                #target_theta = lim_steer
                #target_rad = lim_steer/180*math.pi
            #self.get_logger().info('ccccccccccccccccc c_obs_near lllllllllllllll')
        elif ~np.any(r_obs) and np.any(l_obs) and np.any(c_obs_near) :
            speed = 0.0
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            if abs(target_theta) < 3:
                speed = -0.15
                #target_theta = -lim_steer
                #target_rad = -lim_steer/180*math.pi
            #self.get_logger().info('ccccccccccccccccc c_obs_near rrrrrrrrrrrrrrr')
        elif np.any(c_obs_near) :
            speed = -0.15
            #self.get_logger().info('ccccccccccccccccc c_obs_near ccccccccccccccc')
        elif np.any(c_obs) :
            speed = 0.25
            #self.get_logger().info('dddddddddd speed down ddddddddddd')
        
        '''
        if self.rh_obs and ~self.lh_obs:
            speed = 0.15
            target_rad = lim_steer/180*math.pi
        elif ~self.rh_obs and self.lh_obs:
            speed = 0.15
            target_rad = -lim_steer/180*math.pi
        '''
        
        
        #elif abs(target_theta)  > 90:
        #    speed = 0.2
        #else:
        #    speed = 0.2
        #self.get_logger().info('speed = %f' % (speed))
        target_theta = target_theta +90/180*math.pi
        target_rad_pd = self.sensim0(target_rad)
        #target_rad_pd = target_rad
        
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
        #self.get_logger().info('Publishing cmd_vel: linear.x = %f, angular.z = %f : %f deg' % (twist_msg.linear.x, twist_msg.angular.z, math.degrees(twist_msg.angular.z)))
        
    def set_target_rad(self, path, position_x, position_y, target_dist, theta_x, theta_y, theta_z):
        path_x_diff = path[0,:] - position_x
        path_y_diff = path[1,:] - position_y
        path_diff = np.sqrt(path_x_diff**2 + path_y_diff**2)
        path_diff_target_dist = np.abs(path_diff - target_dist)
        path_target_ind_sort = np.argsort(path_diff_target_dist)[:4] #check 4point
        target_ind = np.max(path_target_ind_sort)
        target_point = path[:,target_ind]
        relative_point_x = target_point[0] - position_x
        relative_point_y = target_point[1] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, target_point[2]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        target_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        target_theta = (target_rad) * (180 / math.pi)
        return target_rad, target_theta
        
        
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
    
    def get_odom_ref(self, msg):
        self.ref_position_x = msg.pose.pose.position.x
        self.ref_position_y = msg.pose.pose.position.y
        self.ref_position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.ref_theta_x = 0 #roll /math.pi*180
        self.ref_theta_y = 0 #pitch /math.pi*180
        self.ref_theta_z = yaw /math.pi*180
        
        if (self.stop_xy[self.stop_num,0] < self.ref_position_x) and (self.ref_position_x < self.stop_xy[self.stop_num,1]) and (self.stop_xy[self.stop_num,2] < self.ref_position_y) and (self.ref_position_y < self.stop_xy[self.stop_num,3]):
            if self.stop_xy[self.stop_num,4] > 0:
                self.get_logger().info('####### stop flag on %f #######' % (self.stop_num))
                self.stop_flag = 1;
            else:
                self.get_logger().info('####### through flag on %f #######' % (self.stop_num))
            self.stop_num = self.stop_num + 1;
            
        
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
        
    def obs_steer(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        #map_obs
        self.obs_points = points
        
        rh_obs = self.pcd_serch(points, -0.2,1,0,0.7)
        if len(rh_obs[0,:]) > 10:
            self.rh_obs = 1
        else:
            self.rh_obs = 0
        lh_obs = self.pcd_serch(points, -0.2,1,-0.7,0)
        if len(lh_obs[0,:]) > 10:
            self.lh_obs = 1
        else:
            self.lh_obs = 0

    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ) )
        pcd_mask = pointcloud[:, pcd_ind]
        return pcd_mask

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
