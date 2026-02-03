import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from geometry_msgs.msg import PolygonStamped, PoseStamped, Pose, Point
from visualization_msgs.msg import Marker

import numpy as np
import cv2
import cupy as cp  # CuPy eklendi

import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

class ParkingArea3D(Node):
    def __init__(self):
        super().__init__("parking_area_3d")

        # --- Camera intrinsics ---
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.width = 1280  # Default, CameraInfo gelince güncellenecek
        self.height = 720  # Default, CameraInfo gelince güncellenecek
        self.is_initialized = False

        # --- Parking Area Polygon (image frame, pixel coordinates) ---
        self.parking_poly_np = None
        self.parking_mask = None # Poligon testi hızlandırmak için maske

        # --- TF Buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(
            CameraInfo,
            "/zed_fixed/camera_info",
            self.camera_info_cb,
            10
        )
        self.create_subscription(
            PolygonStamped,
            "/parking_area/polygon",
            self.polygon_cb,
            10
        )
        self.center_pose_pub = self.create_publisher(
            PoseStamped,
            "/parking_area/center_pose",
            5
        )
                
        self.create_subscription(
            PointCloud2,
            "/velodyne_points",
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
            "3D Parking Area Node (CuPy Accelerated) started."
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
            self.get_logger().info(
                f"Kamera intrinsics alındı: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
            )

    def polygon_cb(self, msg: PolygonStamped):
        points = np.array(
            [[point.x, point.y] for point in msg.polygon.points],
            dtype=np.int32
        )
        self.parking_poly_np = points.astype(np.float32)

        # Optimization: Poligonu bir kez maske olarak çiziyoruz (CPU'da hızlıdır)
        # Bu maske, her nokta için pointPolygonTest yapmaktan çok daha hızlıdır.
        if self.width is not None and self.height is not None:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            self.parking_mask = cp.asarray(mask) # Maskeyi GPU'ya gönder

    def lidar_cb(self, msg: PointCloud2):
        if not self.is_initialized or self.parking_poly_np is None or self.parking_mask is None:
            return

        # 1. TF LOOKUPS (CPU)
        try:
            # Lidar -> Camera
            T_lidar_to_cam = self.tf_buffer.lookup_transform(
                "zed2_left_camera_optical_frame", "ouster", rclpy.time.Time()
            )
            # Lidar -> Base
            T_lidar_to_base = self.tf_buffer.lookup_transform(
                "base_link", "ouster", rclpy.time.Time()
            )
            # Camera -> Lidar (For Ray Tracing)
            T_cam_to_lidar = self.tf_buffer.lookup_transform(
                "ouster", "zed2_left_camera_optical_frame", rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # 2. DATA PREPARATION (CPU -> GPU)
        # PointCloud verisini hızlıca numpy array'e çeviriyoruz
        # Not: pc2.read_points bir generator'dır, listeye çevirip numpy yapmak gerekir.
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_list = [[p[0], p[1], p[2]] for p in gen]
        points_np = np.array(points_list, dtype=np.float32)

        if points_np.shape[0] == 0:
            return

        # Veriyi GPU'ya taşı
        points_lidar_gpu = cp.asarray(points_np) # (N, 3)

        # 3. MATRICES TO GPU
        # Rotation / Translation matrislerini hazırla
        R_l2c_gpu = cp.asarray(self.quat_to_mat(T_lidar_to_cam.transform.rotation))
        t_l2c_gpu = cp.asarray([T_lidar_to_cam.transform.translation.x,
                                T_lidar_to_cam.transform.translation.y,
                                T_lidar_to_cam.transform.translation.z])

        R_l2b_gpu = cp.asarray(self.quat_to_mat(T_lidar_to_base.transform.rotation))
        t_l2b_gpu = cp.asarray([T_lidar_to_base.transform.translation.x,
                                T_lidar_to_base.transform.translation.y,
                                T_lidar_to_base.transform.translation.z])
        
        R_c2l_gpu = cp.asarray(self.quat_to_mat(T_cam_to_lidar.transform.rotation))

        # --- PART 1: FILTERING POINTS (GPU Accelerated) ---
        
        # Lidar -> Camera Transform: P_cam = R * P_lidar + t
        # Matris çarpımı için transpoze alıyoruz: (N,3) @ (3,3).T
        points_cam_gpu = points_lidar_gpu @ R_l2c_gpu.T + t_l2c_gpu

        # X, Y, Z ayrıştır
        X = points_cam_gpu[:, 0]
        Y = points_cam_gpu[:, 1]
        Z = points_cam_gpu[:, 2]

        # Sadece Z > 0 olanlar (Kamera önündekiler)
        valid_z_mask = Z > 0.0

        # Projeksiyon (Vectorized): u = fx * X/Z + cx
        # Sadece valid Z noktaları için hesaplama yapıp array boyutunu korumak yerine 
        # tüm matris üzerinde işlem yapıp maskeyi sonra uygulamak CuPy'de genelde daha hızlıdır.
        # Sıfıra bölmeyi önlemek için küçük bir epsilon veya maske kullanabiliriz ama 
        # zaten Z>0 maskesiyle filtreleyeceğiz.
        
        u = (self.fx * (X / Z) + self.cx).astype(cp.int32)
        v = (self.fy * (Y / Z) + self.cy).astype(cp.int32)

        # Görüntü sınırları kontrolü
        in_bounds_mask = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        
        # Nihai geçerli nokta maskesi (Z > 0 VE Ekran içinde)
        final_candidate_mask = valid_z_mask & in_bounds_mask
        
        # İndisleri al (bu aşamada veri sayısını azaltıyoruz)
        valid_indices = cp.where(final_candidate_mask)[0]
        
        if valid_indices.size == 0:
            return

        u_valid = u[valid_indices]
        v_valid = v[valid_indices]

        # Poligon Testi (GPU Mask Indexing)
        # parking_mask'tan [v, u] değerlerini okuyoruz. Değer > 0 ise poligondadır.
        in_poly_mask = self.parking_mask[v_valid, u_valid] > 0

        # Poligon içindeki orijinal lidar noktalarının indeksleri
        final_indices = valid_indices[in_poly_mask]
        
        if final_indices.size == 0:
            return

        # Lidar -> Base Link Transform (Seçili noktalar için)
        points_selected_lidar = points_lidar_gpu[final_indices]
        points_base_gpu = points_selected_lidar @ R_l2b_gpu.T + t_l2b_gpu

        # --- PART 2: POLYGON CENTER CALCULATION (GPU Accelerated) ---
        
        poly_center_accum = []
        
        # Poligon köşe noktalarını (pixel) al
        poly_pts_cpu = self.parking_poly_np # Numpy array (M, 2)
        
        # Her bir köşe noktası için:
        for pt in poly_pts_cpu:
            u_p, v_p = pt[0], pt[1]
            
            # 2D -> 3D Ray (Camera Frame)
            x_norm = (u_p - self.cx) / self.fx
            y_norm = (v_p - self.cy) / self.fy
            
            # Ray vector (normalized)
            ray_dir_cam = np.array([x_norm, y_norm, 1.0], dtype=np.float32)
            ray_dir_cam /= np.linalg.norm(ray_dir_cam)
            
            # Camera -> Lidar Ray Transform
            ray_dir_cam_gpu = cp.asarray(ray_dir_cam)
            ray_dir_lidar_gpu = R_c2l_gpu @ ray_dir_cam_gpu # (3,) vector

            # Tüm Lidar noktaları ile Ray arasındaki izdüşümü hesapla
            # proj = P . ray
            # points_lidar_gpu: (N, 3), ray_dir_lidar_gpu: (3,)
            proj = points_lidar_gpu @ ray_dir_lidar_gpu
            
            mask = proj > 0
            if not cp.any(mask):
                continue
            
            pts_masked = points_lidar_gpu[mask]
            proj_masked = proj[mask]
            
            # Ray üzerindeki en yakın nokta: closest = proj * ray_dir
            closest_points = proj_masked[:, None] * ray_dir_lidar_gpu[None, :]
            
            # Lidar noktası ile Ray üzerindeki izdüşüm arasındaki mesafe
            # dist = || P - closest ||
            diff = pts_masked - closest_points
            dists = cp.linalg.norm(diff, axis=1)
            
            # En küçük mesafeye sahip olan noktanın indeksi
            idx = cp.argmin(dists)
            p_lidar_hit = pts_masked[idx]
            
            # Lidar -> Base Link
            p_base_hit = R_l2b_gpu @ p_lidar_hit + t_l2b_gpu
            poly_center_accum.append(p_base_hit)

        if not poly_center_accum:
            return

        # Ortalama hesapla
        poly_center_stack = cp.stack(poly_center_accum)
        poly_center_gpu = cp.mean(poly_center_stack, axis=0)
        
        # Sonuçları CPU'ya (Numpy/Float) geri çek
        poly_center = cp.asnumpy(poly_center_gpu) # array([x, y, z])

        # Publish
        center_point = Point()
        center_point.x = float(poly_center[0])
        center_point.y = float(poly_center[1])
        center_point.z = float(poly_center[2])
        self.center_point_pub.publish(center_point)
        
        self.publish_center_pose_map(poly_center)

    def quat_to_mat(self, q):
        """Quaternion -> 3x3 Rotation Matrix (Numpy versiyonu korunabilir, sadece veri üretir)"""
        w, x, y, z = q.w, q.x, q.y, q.z
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R

    def publish_center_pose_map(self, p_base):
        # Bu fonksiyon TF ve mesaj oluşturma içerdiği için CPU'da kalması daha uygundur.
        pose_base = Pose()
        pose_base.position.x = float(p_base[0])
        pose_base.position.y = float(p_base[1])
        pose_base.position.z = float(p_base[2])
        pose_base.orientation.w = 1.0

        try:
            T_base_to_map = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            pose_map = do_transform_pose(pose_base, T_base_to_map)
        except Exception as e:
            self.get_logger().warn(f"TF base_link->map failed: {e}")
            return

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose = pose_map

        self.center_pose_pub.publish(pose_msg)

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