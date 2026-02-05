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
        self.current_confidence = 0.0
        
        # --- Orientation Smoothing ---
        self.prev_yaw = None
        self.yaw_history = []
        self.max_history_size = 5
        self.yaw_change_threshold = np.pi / 3  # 60 degrees - more tolerant for side approaches
        
        # --- Position Locking (orientation stays dynamic) ---
        self.locked_position_map = None  # Only lock position, not orientation
        self.lock_distance_threshold = 8.0
        self.is_position_locked = False
        
        # --- ORIENTATION OFFSET ---
        # Manuel düzeltme için: 0, 90, -90, 180 derece deneyin
        self.declare_parameter('orientation_offset_degrees', 0.0)
        self.orientation_offset = np.radians(
            self.get_parameter('orientation_offset_degrees').value
        )
        
        # PCA eksen seçimi: 'major' veya 'minor'
        self.declare_parameter('pca_axis', 'minor')  # 'major' veya 'minor'
        self.pca_axis_mode = 'major'

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
        self.pose_debug_pub = self.create_publisher(PoseStamped, "/parking_area/pose_debug", 10)
        
        self.create_subscription(
            GaeShellPolygon,
            "/parking_area/polygon",
            self.polygon_cb,
            10
        )
        
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
            f"3D Parking Area Node started with:\n"
            f"  - PCA Axis: {self.pca_axis_mode}\n"
            f"  - Orientation Offset: {np.degrees(self.orientation_offset):.1f}°\n"
            f"  - Lock Distance: {self.lock_distance_threshold}m"
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

    def compute_center_xy_minarearect(self, points_base_ground):
        pts = cp.asnumpy(points_base_ground[:, :2]).astype(np.float32)
        if pts.shape[0] < 20:
            return None

        # outlier kırp (median etrafı)
        med = np.median(pts, axis=0)
        d = np.linalg.norm(pts - med, axis=1)
        thr = np.percentile(d, 70)  # 70-80 arası denenir
        pts = pts[d < thr]
        if pts.shape[0] < 20:
            return None

        hull = cv2.convexHull(pts)  # (N,1,2)
        rect = cv2.minAreaRect(hull)  # ((cx,cy),(w,h),angle)
        (cx, cy), (w, h), ang = rect
        return float(cx), float(cy)


    def polygon_cb(self, msg: GaeShellPolygon):
        self.current_confidence = msg.confidence
        
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
                "zed_left_camera_optical_frame", "ouster", rclpy.time.Time()
            )
            T_lidar_to_base = self.tf_buffer.lookup_transform(
                "base_link", "ouster", rclpy.time.Time()
            )
            T_cam_to_lidar = self.tf_buffer.lookup_transform(
                "ouster", "zed_left_camera_optical_frame", rclpy.time.Time()
            )
        except Exception as e:
            # self.get_logger().warn(f"TF lookup failed: {e}")
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
        
        # --- POINT FILTERING PIPELINE ---
        points_cam_gpu = points_lidar_gpu @ R_l2c_gpu.T + t_l2c_gpu

        X = points_cam_gpu[:, 0]
        Y = points_cam_gpu[:, 1]
        Z = points_cam_gpu[:, 2]

        valid_z_mask = Z > -0.5

        Z_safe = cp.maximum(Z, 0.1)
        u = (self.fx * (X / Z_safe) + self.cx).astype(cp.int32)
        v = (self.fy * (Y / Z_safe) + self.cy).astype(cp.int32)

        in_bounds_mask = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        
        # final_candidate_mask = valid_z_mask & in_bounds_mask
        
        # valid_indices = cp.where(final_candidate_mask)[0]
        
        # if valid_indices.size == 0:
        #     return

        # candidates_indices = valid_indices 

        final_candidate_mask = valid_z_mask & in_bounds_mask
        cand_idx = cp.where(final_candidate_mask)[0]
        if cand_idx.size == 0:
            return

        # u,v değerlerini candidate indekslerine indir
        u_c = u[final_candidate_mask]
        v_c = v[final_candidate_mask]

        # maskeden oku (v,u!)
        mask_vals = self.parking_mask[v_c, u_c]  # cupy array
        in_poly_mask = mask_vals > 0

        candidates_indices = cand_idx[in_poly_mask]



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

        # --- CENTER & ORIENTATION CALCULATION ---
        xy = self.compute_center_xy_minarearect(points_base_ground)
        if xy is not None:
            cx, cy = xy
            cz = float(cp.median(points_base_ground[:, 2]))  # z için median daha stabil
            poly_center = np.array([cx, cy, cz], dtype=np.float32)
        else:
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
        """Publish parking center with DYNAMIC orientation but optionally locked position"""
        
        # Calculate distance to parking spot
        distance_to_spot = np.sqrt(p_base[0]**2 + p_base[1]**2)
        
        # ALWAYS calculate current orientation dynamically from point cloud
        raw_yaw = self.calculate_orientation_pca(points_base_ground)
        smoothed_yaw = self.smooth_orientation(raw_yaw)
        
        # Create pose in base_link frame with DYNAMIC orientation
        pose_base = Pose()
        pose_base.position.x = float(p_base[0])
        pose_base.position.y = float(p_base[1])
        pose_base.position.z = float(p_base[2])
        
        # Use the calculated orientation
        quat = self.yaw_to_quaternion(smoothed_yaw)
        pose_base.orientation.x = quat[0]
        pose_base.orientation.y = quat[1]
        pose_base.orientation.z = quat[2]
        pose_base.orientation.w = quat[3]

        # Debug pose in base_link
        pose_stamped_base = PoseStamped()
        pose_stamped_base.header.frame_id = "base_link"
        pose_stamped_base.header.stamp = self.get_clock().now().to_msg()
        pose_stamped_base.pose = pose_base
        self.pose_debug_pub.publish(pose_stamped_base)
        
        try:
            # Transform to MAP frame
            T_base_to_map = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            pose_map = do_transform_pose(pose_base, T_base_to_map)
        except Exception as e:
            # self.get_logger().warn(f"TF base_link->map failed: {e}")
            return
        
        # POSITION LOCKING (orientation remains dynamic)
        if distance_to_spot < self.lock_distance_threshold:
            if not self.is_position_locked:
                # Lock ONLY the position in MAP frame
                self.locked_position_map = [
                    pose_map.position.x,
                    pose_map.position.y,
                    pose_map.position.z
                ]
                self.is_position_locked = True
                self.get_logger().info(f"🔒 POSITION LOCKED at {distance_to_spot:.2f}m (orientation stays dynamic)")
            
            # Use locked position but current orientation
            pose_to_publish = Pose()
            pose_to_publish.position.x = self.locked_position_map[0]
            pose_to_publish.position.y = self.locked_position_map[1]
            pose_to_publish.position.z = self.locked_position_map[2]
            pose_to_publish.orientation = pose_map.orientation  # Dynamic!
        else:
            # Far from spot - everything updates normally
            if self.is_position_locked:
                self.get_logger().info(f"🔓 POSITION UNLOCKED (distance: {distance_to_spot:.2f}m)")
                self.is_position_locked = False
                self.locked_position_map = None
            
            pose_to_publish = pose_map
        
        # Publish final pose
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose = pose_to_publish

        detection_msg = GaeCamShellDetection()
        detection_msg.center_pose = pose_msg
        detection_msg.confidence = self.current_confidence
        detection_msg.header = pose_msg.header

        self.center_pose_pub.publish(detection_msg)
        
        # Log final orientation
        yaw_deg = np.degrees(smoothed_yaw)
        self.get_logger().info(
            f"📍 Final Orientation: {yaw_deg:.1f}° | Distance: {distance_to_spot:.2f}m | "
            f"Locked: {self.is_position_locked}",
            throttle_duration_sec=1.0
        )

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion (rotation around z-axis)"""
        return [
            0.0,  # x
            0.0,  # y
            np.sin(yaw / 2.0),  # z
            np.cos(yaw / 2.0)   # w
        ]
        
    def calculate_orientation_pca(self, points_base_ground):
        """Calculate parking area orientation using PCA on ground points"""
        points_cpu = cp.asnumpy(points_base_ground[:, :2])  # Only x,y
        
        if points_cpu.shape[0] < 10:
            return self.prev_yaw if self.prev_yaw is not None else 0.0
        
        # Center the points
        centered = points_cpu - np.mean(points_cpu, axis=0)
        
        # Covariance matrix and eigenvector analysis
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvalues and eigenvectors (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate aspect ratio for debugging
        aspect_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-6)
        
        # Select axis based on parameter
        if self.pca_axis_mode == 'minor':
            # Use minor axis (second eigenvector - perpendicular to spread)
            direction = eigenvectors[:, 1]
            axis_name = "MINOR (perpendicular)"
        else:
            # Use major axis (first eigenvector - along spread)
            direction = eigenvectors[:, 0]
            axis_name = "MAJOR (parallel)"
        
        # Calculate yaw angle from selected direction
        yaw = np.arctan2(direction[1], direction[0])
        
        # Apply manual offset
        yaw = self.normalize_angle(yaw + self.orientation_offset)
        
        # Debug info
        self.get_logger().info(
            f"PCA: {axis_name} | Ratio: {aspect_ratio:.2f} | "
            f"Raw yaw: {np.degrees(yaw - self.orientation_offset):.1f}° | "
            f"Offset: {np.degrees(self.orientation_offset):.1f}° | "
            f"Final: {np.degrees(yaw):.1f}°",
            throttle_duration_sec=2.0
        )
        
        # Resolve 180-degree ambiguity using previous orientation
        if self.prev_yaw is not None:
            yaw_alt = yaw + np.pi if yaw < 0 else yaw - np.pi
            diff1 = abs(self.normalize_angle(yaw - self.prev_yaw))
            diff2 = abs(self.normalize_angle(yaw_alt - self.prev_yaw))
            if diff2 < diff1:
                yaw = yaw_alt
                self.get_logger().info("180° flip applied", throttle_duration_sec=3.0)
        
        return yaw
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def smooth_orientation(self, new_yaw):
        """Smooth orientation using circular moving average with jump detection"""
        
        if self.prev_yaw is None:
            self.prev_yaw = new_yaw
            self.yaw_history.append(new_yaw)
            return new_yaw
        
        # Check for large jumps
        diff = self.normalize_angle(new_yaw - self.prev_yaw)
        
        if abs(diff) > self.yaw_change_threshold:
            # self.get_logger().warn(
            #     f"⚠️ Large orientation jump: {np.degrees(diff):.1f}° - ignoring",
            #     throttle_duration_sec=2.0
            # )
            return self.prev_yaw
        
        # Add to history
        self.yaw_history.append(new_yaw)
        
        if len(self.yaw_history) > self.max_history_size:
            self.yaw_history.pop(0)
        
        # Circular mean for angles (proper averaging for angles)
        if len(self.yaw_history) > 1:
            sin_sum = sum(np.sin(y) for y in self.yaw_history)
            cos_sum = sum(np.cos(y) for y in self.yaw_history)
            smoothed_yaw = np.arctan2(sin_sum, cos_sum)
        else:
            smoothed_yaw = new_yaw
        
        self.prev_yaw = smoothed_yaw
        return smoothed_yaw

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