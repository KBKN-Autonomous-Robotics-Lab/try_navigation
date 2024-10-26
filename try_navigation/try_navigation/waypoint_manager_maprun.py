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
import yaml
import os
import time
import geometry_msgs.msg as geometry_msgs
import glob
import cv2
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import transforms3d
from geometry_msgs.msg import Quaternion

# C++と同じく、Node型を継承します。
class WaypointManagerMaprun(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('waypoint_manager_maprun_node')
        
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
        
        map_qos_profile_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth = 10
        )
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_wheel', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(OccupancyGrid,'/reflect_map_local', self.get_reflect_map_local, map_qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # タイマーを0.1秒（100ミリ秒）ごとに呼び出す
        self.timer = self.create_timer(0.1, self.waypoint_manager)
        
        # Publisherを作成
        self.current_waypoint_publisher = self.create_publisher(geometry_msgs.PoseArray, 'current_waypoint', qos_profile) #set publish pcd topic name
        self.map_match_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_local', map_qos_profile_sub)
        self.map_match_ref_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_ref', map_qos_profile_sub)
        self.map_match_result_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_result', map_qos_profile_sub)
        #self.map_match_result_publisher = self.create_publisher(sensor_msgs.Image, 'reflect_map_match_result', map_qos_profile_sub)
        #self.bridge = CvBridge()
        self.odom_ref_slam_publisher = self.create_publisher(nav_msgs.Odometry, 'odom_ref_slam', qos_profile)
        self.waypoint_path_publisher = self.create_publisher(nav_msgs.Path, 'waypoint_path', qos_profile) 
        
        #パラメータ
        #waypoint init
        self.current_waypoint = 0
        self.stop_flag = 0
        
        #positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        self.odom_x_buff = 0
        self.odom_y_buff = 0
        self.ref_slam_x_buff = 0
        self.ref_slam_y_buff = 0
        self.ref_slam_diff = [0,0,0]
        
        # Waypoint YAMLファイルを読み込む
        #Global reflect map load: PGM、JPEG、YAMLファイルを行列形式で読み込み
        map_base_name = "waypoint_map" # 対象画像の名前
        #folder_path = os.path.expanduser('~/ros2_ws/src/map/nakaniwa')
        folder_path = os.path.expanduser('~/ros2_ws/src/map/tukuba_kakunin')
        # フォルダ内のpgmファイルを検索
        pgm_files = glob.glob(os.path.join(folder_path, '*.pgm'))
        pgm_file_count = len(pgm_files)
        map_file_path = os.path.join(folder_path, map_base_name) # パスの連結
        
        global_reflect_map_pgms = []
        map_resolution = []; map_origin=[]; map_occupied_thresh=[];map_free_thresh=[];
        reflect_map_obs_matrices = []
        for map_number in range(pgm_file_count):
            map_number_str = str(map_number).zfill(3)
            #load pgm
            pgm_filename = os.path.join(folder_path, f'{map_file_path}_{map_number_str}' + ".pgm")
            print(f"pgm_filename: {pgm_filename}")
            global_reflect_map_pgm = cv2.imread(pgm_filename, cv2.IMREAD_GRAYSCALE)
            global_reflect_map_pgms.append(global_reflect_map_pgm)
            #load yaml
            yaml_filename = os.path.join(folder_path, f'{map_file_path}_{map_number_str}' + ".yaml")
            with open(yaml_filename, 'r') as yaml_file:
                map_yaml_data = yaml.safe_load(yaml_file)
            map_resolution.append(map_yaml_data['resolution'])
            map_origin.append(map_yaml_data['origin'])
            map_occupied_thresh.append(map_yaml_data['occupied_thresh'])
            map_free_thresh.append(map_yaml_data['free_thresh'])
            #load jpeg
            jpeg_filename = os.path.join(folder_path, f'{map_file_path}_{map_number_str}' + ".jpeg")
            reflect_map_obs = cv2.imread(jpeg_filename)
            red_judge1 = reflect_map_obs[:,:,0] < 100
            red_judge2 = reflect_map_obs[:,:,2] > 200
            reflect_map_obs_data = red_judge1 * red_judge2 * 100
            reflect_map_obs_matrices.append(reflect_map_obs_data)
            
        self.global_reflect_map_pgm = np.stack(global_reflect_map_pgms)
        self.global_reflect_map_resolution = np.stack(map_resolution)
        self.global_reflect_map_origin = np.stack(map_origin)
        self.global_reflect_map_occupied_thresh = np.stack(map_occupied_thresh)
        self.global_reflect_map_free_thresh = np.stack(map_free_thresh)
        self.reflect_map_obs_matrices = np.stack(reflect_map_obs_matrices)
        
        #print(f"self.global_reflect_map_pgm: {self.global_reflect_map_pgm.shape}")
        #print(f"self.global_reflect_map_pgm: {self.global_reflect_map_pgm.shape[0]}")
        #print(f"self.global_reflect_map_pgm[0]: {self.global_reflect_map_pgm[0]}")
        #print(f"self.global_reflect_map_resolution[0]: {self.global_reflect_map_resolution[0]}")
        #print(f"self.global_reflect_map_origin[0]: {self.global_reflect_map_origin[0]}")
        #print(f"self.reflect_map_obs_matrices[0]: {self.reflect_map_obs_matrices[0]}")
        
        wp_x = []; wp_y=[]; wp_z=[];
        for wp_number in range(self.global_reflect_map_pgm.shape[0]-1):
            x_offset = (len(self.global_reflect_map_pgm[wp_number][0]) * self.global_reflect_map_resolution[0])/2
            y_offset = (len(self.global_reflect_map_pgm[wp_number][1]) * self.global_reflect_map_resolution[0])/2
            x = self.global_reflect_map_origin[wp_number][0] + x_offset #pgm end + (pgm len * grid)/2
            y = self.global_reflect_map_origin[wp_number][1] + y_offset #pgm end + (pgm len * grid)/2
            z = self.global_reflect_map_origin[wp_number][2]
            wp_x.append(x)
            wp_y.append(y)
            wp_z.append(z)
        self.waypoints = np.array([wp_x,wp_y,wp_z])
        print(f"self.waypoints ={self.waypoints}")
        print(f"self.waypoints ={self.waypoints.shape}")
        #############################
        '''
        waypoint_map_yaml_path_name = "kbkn_maps/waypoints/hosei/m2/courtyard_Senior.yaml" # waypoint mapの名前
        py_path = "/home/ubuntu/ros2_ws/src/"#os.path.dirname(os.path.abspath(__file__)) # 実行ファイルのディレクトリ名
        waypoint_map_yaml_file_path = os.path.join(py_path, waypoint_map_yaml_path_name) # パスの連結
        print(f"waypoint_map_yaml_path_name ={waypoint_map_yaml_path_name}")
        print(f"py_path ={py_path}")
        
        with open(waypoint_map_yaml_file_path, 'r') as yaml_file:
            waypoint_map_yaml_data = yaml.safe_load(yaml_file)
        #print(f"waypoint_map_yaml_data ={waypoint_map_yaml_data}")
        waypoints = waypoint_map_yaml_data['waypoints']
        x = [point['point']['x'] for point in waypoints]
        y = [point['point']['y'] for point in waypoints]
        z = [point['point']['z'] for point in waypoints]
        self.waypoints = np.array([x, y, z])
        print(f"self.waypoints ={self.waypoints}")
        print(f"self.waypoints ={self.waypoints.shape}")
        '''
    def waypoint_manager(self):
        self.time_stamp = self.get_clock().now().to_msg()
        #self.get_logger().info('waypoint manager cntl')
        position_x=self.position_x; position_y=self.position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        
        #waypoint theta & dist
        set_waypoint = self.waypoints[:,self.current_waypoint] - self.ref_slam_diff
        
        relative_point_x = set_waypoint[0] - position_x
        relative_point_y = set_waypoint[1] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, self.waypoints[2,self.current_waypoint]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        waypoint_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        waypoint_dist = math.sqrt(relative_point_x**2 + relative_point_y**2)
        waypoint_theta = abs(waypoint_rad * (180 / math.pi))
        
        #set judge dist
        if abs(waypoint_theta) > 90:
            determine_dist = 8
        else:
            determine_dist = 3
        #check if the waypoint reached
        if waypoint_dist < determine_dist:
            self.current_waypoint += 1
            if self.current_waypoint > (len(self.waypoints[0,:]) - 1):
                self.stop_flag = 1
                self.get_logger().info('GOAL : stop_flag = %f' % (self.stop_flag))
        #self.get_logger().info('current_waypoint:x = %f, y = %f : waypoint_no = %f' % (self.waypoints[0,self.current_waypoint], self.waypoints[1,self.current_waypoint], self.current_waypoint))
        
        
        #publish
        pose_array = self.current_waypoint_msg(set_waypoint, 'odom')
        self.current_waypoint_publisher.publish(pose_array)
        
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
        
    def get_reflect_map_local(self, msg):
        t_stamp = msg.header.stamp
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))

        # 範囲外の値を修正
        data[data == -1] = 255  # 未探索セルを白に設定
        data[data == 100] = 0   # 障害物セルを黒に設定
        data[data == 0] = 200   # 空きセルをグレーに設定
        flipped_data = cv2.flip(data, 0, dst=None)

        # NumPy配列をJPEGフォーマットにエンコード
        _, buffer = cv2.imencode('.jpg', flipped_data)

        # バッファをデコードして画像データとして読み込む
        reflect_map_local = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if reflect_map_local is not None:
            print(f"Image read successfully with shape: {reflect_map_local.shape}")
        else:
            print("Failed to read the image.")
        
        position_x=self.position_x; position_y=self.position_y; position_z=self.position_z;
        position = np.array([position_x, position_y, position_z])
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        map_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        MAP_RANGE = 10.0
        ground_pixel = 1000/50
        self.map_data_local = make_map_msg(reflect_map_local, ground_pixel, position, map_orientation, t_stamp, MAP_RANGE, "odom")
        self.map_match_local_publisher.publish(self.map_data_local)
        
        #global map option
        map_global_data_set = np.array(100-(self.global_reflect_map_pgm[self.current_waypoint])/255*100, dtype='i8')
        x_offset = (len(self.global_reflect_map_pgm[self.current_waypoint][0]) * self.global_reflect_map_resolution[0])/2
        y_offset = (len(self.global_reflect_map_pgm[self.current_waypoint][1]) * self.global_reflect_map_resolution[0])/2
        map_x = self.global_reflect_map_origin[self.current_waypoint][0] + x_offset #pgm end + (pgm len * grid)/2
        map_y = self.global_reflect_map_origin[self.current_waypoint][1] + y_offset #pgm end + (pgm len * grid)/2
        position_map = np.array([map_x, map_y, 0.0])
        MAP_RANGE_GL = x_offset
        map_ground_pixel = float(1.0/self.global_reflect_map_resolution[self.current_waypoint])
        #print(f"position_map: {position_map}")
        #print(f"MAP_RANGE_GL: {MAP_RANGE_GL}")
        #print(f"map_ground_pixel: {map_ground_pixel}")
        '''
        temp_pixel = self.global_reflect_map_resolution[self.current_waypoint]
        map_origin = self.global_reflect_map_origin[self.current_waypoint]
        clip_size = (MAP_RANGE + 5.0) 
        print(f"temp_pixel: {temp_pixel}")
        print(f"map_origin: {map_origin}")
        print(f"clip_size: {clip_size}")
        clip_min_x = round( ( position[0] - map_origin[0] -position_map[0]  - clip_size) / temp_pixel)
        clip_max_x = round( ( position[0] - map_origin[0] -position_map[0]  + clip_size) / temp_pixel)
        clip_min_y = round( ( position[1] - map_origin[1] -position_map[1]  - clip_size) / temp_pixel)
        clip_max_y = round( ( position[1] - map_origin[1] -position_map[1]+ clip_size) / temp_pixel)
        print(f"position_map: {position_map}")
        print(f"position: {position}")
        print(f"clip xy: {clip_min_x, clip_max_x, clip_min_y, clip_max_y}")
        #clip_map = self.global_reflect_map_pgm[clip_min_y:clip_max_y, clip_min_x:clip_max_x]
        #clip_map = map_global_data_set[clip_min_y:clip_max_y, clip_min_x:clip_max_x]
        clip_map = map_global_data_set[clip_min_y:clip_max_y, clip_min_x:clip_max_x].astype(np.uint8)
        #clip_map = map_global_data_set[clip_min_y:clip_min_y+2*clip_size/temp_pixel, clip_min_x:clip_min_x+2*clip_size/temp_pixel].astype(np.uint8)
        #clip_map = map_global_data_set[0:500, 0:500].astype(np.uint8)
        self.map_data_global = make_map_msg(clip_map, map_ground_pixel, position_map, map_orientation, t_stamp, clip_size, "odom")
        '''
        self.map_data_global = make_map_msg(map_global_data_set, map_ground_pixel, position_map, map_orientation, t_stamp, MAP_RANGE_GL, "odom")
        self.map_match_ref_publisher.publish(self.map_data_global)
        
        #matching
        temp_result = cv2.matchTemplate(map_global_data_set.astype(np.uint8), reflect_map_local, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_result)
        match_percentage = max_val * 100
        print(f"Max match value: {match_percentage:.2f}%")
        print(f"Match location: {max_loc}")
        max_loc_x = max_loc[0]/map_ground_pixel
        max_loc_y = max_loc[1]/map_ground_pixel
        max_loc_position_x = position_map[0] - MAP_RANGE_GL + max_loc_x + MAP_RANGE
        max_loc_position_y = position_map[1] + MAP_RANGE_GL - max_loc_y - MAP_RANGE
        print(f"Match max_loc x,y: { max_loc_x, max_loc_y}")
        print(f"robot position: { position}")
        print(f"map position_map: { position_map}")
        print(f"max_loc_position: { max_loc_position_x, max_loc_position_y}")
        ref_slam_diff_x = max_loc_position_x - self.ref_slam_x_buff
        ref_slam_diff_y = max_loc_position_y - self.ref_slam_y_buff
        ref_slam_diff = math.sqrt(ref_slam_diff_x**2 + ref_slam_diff_y**2 )
        
        if (match_percentage > 30) and (ref_slam_diff < 0.5):
            ref_slam_x = max_loc_position_x
            ref_slam_y = max_loc_position_y
            ref_slam_xy = [ref_slam_x, ref_slam_y, position_z]
            temp_result_image_set = map_global_data_set[max_loc[1]:max_loc[1]+len(reflect_map_local[:,0]), max_loc[0]:max_loc[0]+len(reflect_map_local[0,:])]
            temp_result_image = make_map_msg(temp_result_image_set, map_ground_pixel, position, map_orientation, t_stamp, MAP_RANGE, "odom")
            self.map_match_result_publisher.publish(temp_result_image)
        else:
            ref_slam_x = self.ref_slam_x_buff + position[0] - self.odom_x_buff
            ref_slam_y = self.ref_slam_y_buff + position[1] - self.odom_y_buff
            print(f"XXXXX NOT Match location XXXXX")
        
        self.odom_x_buff = position[0]
        self.odom_y_buff = position[1]
        self.ref_slam_x_buff = ref_slam_x
        self.ref_slam_y_buff = ref_slam_y
        
        self.ref_slam_diff = [ref_slam_x - position[0], ref_slam_y - position[1], 0]
        
        #publish
        odom_ref_slam_msg = odometry_msg(ref_slam_x, ref_slam_y, position_z, theta_x, theta_y, theta_z, t_stamp, 'odom')
        self.odom_ref_slam_publisher.publish(odom_ref_slam_msg)
        waypoint_path = path_msg(self.waypoints, t_stamp, 'odom')
        self.waypoint_path_publisher.publish(waypoint_path) 
            
        '''
        # グレースケール画像の型をCV_8Uに変換
        img1 = cv2.normalize(map_global_data_set, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img2 = cv2.normalize(reflect_map_local, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        angle, tx, ty, good_matches, keypoints1, keypoints2 = self.match_images(img1, img2)
        if angle is not None and tx is not None and ty is not None:
            print(f"Rotation angle: {angle:.2f} degrees")
            print(f"Translation: x = {tx:.2f}, y = {ty:.2f}")
            # マッチング結果の画像をパブリッシュ
            matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            ros_image = self.bridge.cv2_to_imgmsg(matches_img, encoding="bgr8")
            self.map_match_result_publisher.publish(ros_image)
            #self.map_match_result_publisher.publish(self.map_data_local)
        '''
        
    def match_images(self, img1, img2, min_matches=10, match_threshold=30):
        
        
        # ORB特徴点検出器の生成
        orb = cv2.ORB_create()

        # 特徴点とディスクリプタの検出
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # BFMatcherを使用して特徴点をマッチング
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
         # マッチング結果が少ない場合は処理を中止
        if len(matches) < min_matches:
            self.get_logger().warn(f"Not enough matches ({len(matches)}/{min_matches}). Skipping.")
            return None, None, None, None, None, None

        # 有効なマッチングのみを抽出
        good_matches = [m for m in matches if m.distance < match_threshold]

        # 有効なマッチングが少ない場合は処理を中止
        if len(good_matches) < min_matches:
            self.get_logger().warn(f"Not enough good matches ({len(good_matches)}/{min_matches}). Skipping.")
            return None, None, None, None, None, None
        
        # マッチングされた特徴点を抽出
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # ホモグラフィ行列の計算
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 回転角度の計算
        angle = -np.degrees(np.arctan2(M[1, 0], M[0, 0]))

        # 平行移動量の計算
        tx = M[0, 2]
        ty = M[1, 2]
        
        return angle, tx, ty, good_matches, keypoints1, keypoints2
    
    def current_waypoint_msg(self, waypoint, set_frame_id):
        pose_array = geometry_msgs.PoseArray()
        pose_array.header.frame_id = set_frame_id
        pose_array.header.stamp = self.time_stamp
        pose = geometry_msgs.Pose()
        pose.position.x = waypoint[0]
        pose.position.y = waypoint[1]
        pose.position.z = waypoint[2]
        
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)
        
        return pose_array


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

def odometry_msg(pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, stamp, frame_id):
    odom_msg = nav_msgs.Odometry()
    odom_msg.header.stamp = stamp
    odom_msg.header.frame_id = frame_id
    
    # 位置情報を設定
    odom_msg.pose.pose.position.x = pos_x 
    odom_msg.pose.pose.position.y = pos_y
    odom_msg.pose.pose.position.z = pos_z
    
    # YawをQuaternionに変換
    roll = theta_x /180*math.pi
    pitch = theta_y /180*math.pi
    yaw = theta_z /180*math.pi
    quat = transforms3d.euler.euler2quat(roll, pitch, yaw)
    odom_msg.pose.pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
    
    return odom_msg


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


def make_map_msg(map_data_set, resolution, position, orientation, header_stamp, map_range, frame_id):
    map_data = OccupancyGrid()
    map_data.header.stamp =  header_stamp
    map_data.info.map_load_time = header_stamp
    map_data.header.frame_id = frame_id
    map_data.info.width = map_data_set.shape[0]
    map_data.info.height = map_data_set.shape[1]
    map_data.info.resolution = 1/resolution #50/1000#resolution
    pos_round = np.round(position * resolution) / resolution
    map_data.info.origin.position.x = float(pos_round[0] -map_range) #位置オフセット
    map_data.info.origin.position.y = float(pos_round[1] -map_range)
    map_data.info.origin.position.z = float(0.0) #position[2]
    map_data.info.origin.orientation.w = float(orientation[0])#
    map_data.info.origin.orientation.x = float(orientation[1])
    map_data.info.origin.orientation.y = float(orientation[2])
    map_data.info.origin.orientation.z = float(orientation[3])
    map_data_cv = cv2.flip(map_data_set, 0, dst = None)
    map_data_int8array = [i for row in  map_data_cv.tolist() for i in row]
    map_data.data = Int8MultiArray(data=map_data_int8array).data
    return map_data
        
# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    waypoint_manager_maprun = WaypointManagerMaprun()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(waypoint_manager_maprun)
    # 明示的にノードの終了処理を行います。
    waypoint_manager_maprun.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
