import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from geometry_msgs.msg import PolygonStamped, PoseStamped, Pose, Point
from visualization_msgs.msg import Marker

import numpy as np
import cv2
import cupy as cp

import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

from gae_msgs.msg import GaeCamShellDetection, GaeShellPolygon

class ParkingArea3D(Node):
    def __init__(self):
        super().__init__("parking_area_3d")

        # --- Camera intrinsics ---
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.width = 1280
        self.height = 720
        self.is_initialized = False

        # --- Parking Area Polygon & Confidence ---
        self.parking_poly_np = None
        self.parking_mask = None
        self.current_confidence = 0.0  # Confidence skorunu saklamak için değişken
        
        # --- Orientation Smoothing ---
        self.prev_yaw = None
        self.yaw_history = []
        self.max_history_size = 5  # Reduced from 10 for faster response
        self.yaw_change_threshold = np.pi / 4  # 45 degrees - more tolerant than 30
        
        # --- Orientation Locking ---
        self.locked_pose_map = None  # Store locked pose in MAP frame (world coordinates)
        self.lock_distance_threshold = 8.0  # FREEZE ZONE: lock at 8m
        self.is_locked = False

        # --- TF Buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(
            CameraInfo,
            "/zed2_left_camera/camera_info",
            self.camera_info_cb,
            10
        )
        
        # [GÜNCELLEME 1] Artık PolygonStamped yerine GaeShellPolygon dinliyoruz
        self.create_subscription(
            GaeShellPolygon,
            "/parking_area/polygon",
            self.polygon_cb,
            10
        )
        

        # [GÜNCELLEME 2] Publisher tipi GaeCamShellDetection olarak değiştirildi
        self.center_pose_pub = self.create_publisher(
            GaeCamShellDetection,
            "/parking_area/center_pose",
            10
        )
                
        self.create_subscription(
            PointCloud2,
            "/ouster/points",
            self.lidar_cb,
            10
        )
        
        self.pc_pub = self.create_publisher(
            PointCloud2,
            "/parking_area/pointcloud3d",
            5
        )
        self.park_pub = self.create_publisher(
            PolygonStamped,
            "/parking_area/transformed_park",
            5
        )
        self.center_point_pub = self.create_publisher(
            Point,
            "/parking_area/center_point",
            5
        )
        self.get_logger().info(
            "3D Parking Area Node (CuPy Accelerated) started with Confidence support."
        )

    def camera_info_cb(self, msg: CameraInfo):
        if not self.is_initialized:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.width = msg.width
            self.height = msg.height
            self.is_initialized = True

    # [GÜNCELLEME 3] Callback artık GaeShellPolygon alıyor
    def polygon_cb(self, msg: GaeShellPolygon):
        # Confidence skorunu sakla
        self.current_confidence = msg.confidence
        
        # İçerideki PolygonStamped mesajına erişim: msg.polygon
        # Onun içindeki noktalara erişim: msg.polygon.polygon.points
        points = np.array(
            [[point.x, point.y] for point in msg.polygon.polygon.points],
            dtype=np.int32
        )
        self.parking_poly_np = points.astype(np.float32)

        if self.width is not None and self.height is not None:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            self.parking_mask = cp.asarray(mask)

    def lidar_cb(self, msg: PointCloud2):
        if not self.is_initialized or self.parking_poly_np is None or self.parking_mask is None:
            return

        # 1. TF LOOKUPS
        try:
            T_lidar_to_cam = self.tf_buffer.lookup_transform(
                "zed2_left_camera_optical_frame", "ouster", rclpy.time.Time()
            )
            T_lidar_to_base = self.tf_buffer.lookup_transform(
                "base_link", "ouster", rclpy.time.Time()
            )
            T_cam_to_lidar = self.tf_buffer.lookup_transform(
                "ouster", "zed2_left_camera_optical_frame", rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # 2. DATA PREPARATION
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_list = [[p[0], p[1], p[2]] for p in gen]
        points_np = np.array(points_list, dtype=np.float32)

        if points_np.shape[0] == 0:
            return

        points_lidar_gpu = cp.asarray(points_np)

        # 3. MATRICES TO GPU
        R_l2c_gpu = cp.asarray(self.quat_to_mat(T_lidar_to_cam.transform.rotation))
        t_l2c_gpu = cp.asarray([T_lidar_to_cam.transform.translation.x,
                                T_lidar_to_cam.transform.translation.y,
                                T_lidar_to_cam.transform.translation.z])

        R_l2b_gpu = cp.asarray(self.quat_to_mat(T_lidar_to_base.transform.rotation))
        t_l2b_gpu = cp.asarray([T_lidar_to_base.transform.translation.x,
                                T_lidar_to_base.transform.translation.y,
                                T_lidar_to_base.transform.translation.z])
        
        # --- PART 1: FILTERING POINTS ---
        points_cam_gpu = points_lidar_gpu @ R_l2c_gpu.T + t_l2c_gpu

        X = points_cam_gpu[:, 0]
        Y = points_cam_gpu[:, 1]
        Z = points_cam_gpu[:, 2]

        valid_z_mask = Z > 0.0

        # Perspective correction
        Z_safe = cp.maximum(Z, 0.1)
        u = (self.fx * (X / Z_safe) + self.cx).astype(cp.int32)
        v = (self.fy * (Y / Z_safe) + self.cy).astype(cp.int32)

        in_bounds_mask = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        
        final_candidate_mask = valid_z_mask & in_bounds_mask
        
        valid_indices = cp.where(final_candidate_mask)[0]
        
        if valid_indices.size == 0:
            return

        u_valid = u[valid_indices]
        v_valid = v[valid_indices]

        # 2D mask check
        in_poly_2d_mask = self.parking_mask[v_valid, u_valid] > 0
        
        candidates_indices = valid_indices[in_poly_2d_mask]
        
        if candidates_indices.size == 0:
            return
            
        # 3D spatial filtering
        points_candidate_cam = points_cam_gpu[candidates_indices]
        Z_candidate = points_candidate_cam[:, 2]
        
        depth_scale = Z_candidate / cp.median(Z_candidate)
        spatial_tolerance = 2.0 * depth_scale
        
        points_candidate_base = points_lidar_gpu[candidates_indices] @ R_l2b_gpu.T + t_l2b_gpu
        median_xy = cp.median(points_candidate_base[:, :2], axis=0)
        
        xy_dist = cp.sqrt(cp.sum((points_candidate_base[:, :2] - median_xy)**2, axis=1))
        spatial_mask = xy_dist < spatial_tolerance.mean()
        
        final_indices = candidates_indices[spatial_mask]
        
        if final_indices.size == 0:
            return

        # Transform to base_link
        points_selected_lidar = points_lidar_gpu[final_indices]
        points_base_gpu = points_selected_lidar @ R_l2b_gpu.T + t_l2b_gpu

        # Adaptive Ground Filtering
        Z_base = points_base_gpu[:, 2]
        z_median = float(cp.median(Z_base))
        
        ground_reference = z_median
        ground_tolerance = 0.5 
        
        ground_height_mask = cp.abs(Z_base - ground_reference) < ground_tolerance
        points_base_ground = points_base_gpu[ground_height_mask]
        
        if points_base_ground.shape[0] < 10:
            points_base_ground = points_base_gpu

        # --- PART 2: CENTER CALCULATION ---
        poly_center_gpu = cp.mean(points_base_ground, axis=0)
        poly_center = cp.asnumpy(poly_center_gpu)
        
        # Publish Debug Point
        center_point = Point()
        center_point.x = float(poly_center[0])
        center_point.y = float(poly_center[1])
        center_point.z = float(poly_center[2])
        self.center_point_pub.publish(center_point)
        
        self.publish_center_pose_map(poly_center, points_base_ground)

    def quat_to_mat(self, q):
        w, x, y, z = q.w, q.x, q.y, q.z
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R

    def publish_center_pose_map(self, p_base, points_base_ground=None):
        """Park alanı merkezini ve oryantasyonunu yayınla"""
        
        # Calculate distance to parking spot (p_base is in base_link frame, vehicle is at origin)
        distance_to_spot = np.sqrt(p_base[0]**2 + p_base[1]**2)
        
        # Calculate current orientation and position
        raw_yaw = self.calculate_orientation_pca(points_base_ground)
        smoothed_yaw = self.smooth_orientation(raw_yaw)
        
        # Create pose in base_link frame
        pose_base = Pose()
        pose_base.position.x = float(p_base[0])
        pose_base.position.y = float(p_base[1])
        pose_base.position.z = float(p_base[2])
        quat = self.yaw_to_quaternion(smoothed_yaw)
        pose_base.orientation.x = quat[0]
        pose_base.orientation.y = quat[1]
        pose_base.orientation.z = quat[2]
        pose_base.orientation.w = quat[3]
        
        try:
            # Transform to MAP frame (world coordinates)
            T_base_to_map = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            pose_map = do_transform_pose(pose_base, T_base_to_map)
        except Exception as e:
            self.get_logger().warn(f"TF base_link->map failed: {e}")
            return
        
        # FREEZE ZONE: Lock pose in MAP frame when within 8 meters
        if distance_to_spot < self.lock_distance_threshold:
            if not self.is_locked:
                # First time entering freeze zone - lock the MAP frame pose
                self.locked_pose_map = pose_map
                self.is_locked = True
                self.get_logger().info(f"🔒 FREEZE ZONE: Locked at {distance_to_spot:.2f}m in MAP frame")
            
            # Use locked pose (in MAP frame - doesn't move with car!)
            pose_to_publish = self.locked_pose_map
        else:
            # Far from spot - update normally
            if self.is_locked:
                self.get_logger().info(f"🔓 FREEZE ZONE RELEASED (distance: {distance_to_spot:.2f}m)")
                self.is_locked = False
                self.locked_pose_map = None
            
            pose_to_publish = pose_map
        
        # Publish the pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose = pose_to_publish
        #         pose_msg.pose.position.x = 33961.55859375
        # pose_msg.pose.position.y = 22.698162078857422
        # pose_msg.pose.position.z = 0.0    
        # pose_msg.pose.orientation.x = 0.0
        # pose_msg.pose.orientation.y = 0.0
        # pose_msg.pose.orientation.z = -0.008741608823815417
        # pose_msg.pose.orientation.w = 0.9999617914076374

        detection_msg = GaeCamShellDetection()
        detection_msg.center_pose = pose_msg
        detection_msg.confidence = self.current_confidence
        detection_msg.header = pose_msg.header

        self.center_pose_pub.publish(detection_msg)

    def yaw_to_quaternion(self, yaw):
        """Yaw açısını quaternion'a çevir (sadece z ekseni etrafında)"""
        return [
            0.0,  # x
            0.0,  # y
            np.sin(yaw / 2.0),  # z
            np.cos(yaw / 2.0)   # w
        ]
        
        
    def calculate_orientation_pca(self, points_base_ground):
        """PCA ile park alanının yönünü hesapla"""
        points_cpu = cp.asnumpy(points_base_ground[:, :2])  # Sadece x,y
        
        if points_cpu.shape[0] < 10:
            # Not enough points, return previous or default
            return self.prev_yaw if self.prev_yaw is not None else 0.0
        
        # Merkeze göre normalize et
        centered = points_cpu - np.mean(points_cpu, axis=0)
        
        # Covariance matrix ve eigenvalue/eigenvector
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # En büyük eigenvalue'ya karşılık gelen eigenvector = ana yön
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Yaw açısını hesapla
        yaw = np.arctan2(main_direction[1], main_direction[0])
        
        # PCA gives ambiguous direction (180 degree ambiguity)
        # Choose the one closer to previous orientation if available
        if self.prev_yaw is not None:
            yaw_alt = yaw + np.pi if yaw < 0 else yaw - np.pi
            if abs(self.normalize_angle(yaw - self.prev_yaw)) > abs(self.normalize_angle(yaw_alt - self.prev_yaw)):
                yaw = yaw_alt
        
        return yaw
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def smooth_orientation(self, new_yaw):
        """Smooth orientation changes using moving average and jump detection"""
        
        # First call initialization
        if self.prev_yaw is None:
            self.prev_yaw = new_yaw
            self.yaw_history.append(new_yaw)
            return new_yaw
        
        # Normalize the difference
        diff = self.normalize_angle(new_yaw - self.prev_yaw)
        
        # Detect large jumps and ignore them
        if abs(diff) > self.yaw_change_threshold:
            self.get_logger().warn(f"Large orientation jump detected: {np.degrees(diff):.1f} degrees, ignoring")
            return self.prev_yaw
        
        # Add to history
        self.yaw_history.append(new_yaw)
        
        # Keep history size limited
        if len(self.yaw_history) > self.max_history_size:
            self.yaw_history.pop(0)
        
        # Circular mean for angles
        if len(self.yaw_history) > 1:
            # Convert to unit vectors, average, then back to angle
            sin_sum = sum(np.sin(y) for y in self.yaw_history)
            cos_sum = sum(np.cos(y) for y in self.yaw_history)
            smoothed_yaw = np.arctan2(sin_sum, cos_sum)
        else:
            smoothed_yaw = new_yaw
        
        self.prev_yaw = smoothed_yaw
        return smoothed_yaw
    
    
    def calculate_orientation_from_polygon(self):
        """Poligonun en uzun kenarından yön hesapla"""
        poly = self.parking_poly_np
        
        max_length = 0
        best_edge = None
        
        # En uzun kenarı bul
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            length = np.linalg.norm(p2 - p1)
            
            if length > max_length:
                max_length = length
                best_edge = (p1, p2)
        
        # Kenar vektöründen yaw hesapla
        edge_vec = best_edge[1] - best_edge[0]
        yaw = np.arctan2(edge_vec[1], edge_vec[0])
        
        return yaw

def main(args=None):
    rclpy.init(args=args)
    node = ParkingArea3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()