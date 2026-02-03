import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch
import numpy as np

class ZedYoloNode(Node):
    def __init__(self):
        super().__init__('zed_yolo_detector')
        
        # 1. Modeli Yükle
        self.get_logger().info("YOLOv8 modeli yükleniyor...")
        self.model = YOLO("bestpro.pt") 
        self.get_logger().info("Model yüklendi!")

        # 2. CvBridge Başlat
        self.bridge = CvBridge()

        # 3. Subscriber (Kamera Görüntüsü)
        self.subscription = self.create_subscription(
            Image,
            '/zed2_left_camera/image_raw',
            self.image_callback,
            10
        )
        
        # 4. Publisher (Park Alanı Poligonu)
        self.poly_pub = self.create_publisher(PolygonStamped, '/parking_area/polygon', 10)

        self.get_logger().info("YOLO Detector Hazır (Approx Kapalı). Topic: /parking_area/polygon")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge Hatası: {e}")
            return

        # Tahmin Yap
        results = self.model(frame, verbose=False)
        result = results[0]
        annotated_frame = frame

        # Maske ve güven skoru kontrolü
        if result.masks is not None and len(result.boxes.conf) > 0:
            
            # --- EN İYİ TESPİTİ SEÇME ---
            scores = result.boxes.conf
            best_index = torch.argmax(scores).item()

            # En iyi maske verisini al (Numpy array: [N, 2])
            best_mask_data = result.masks.xy[best_index]
            
            # --- POLIGON MESAJINI OLUŞTUR VE YAYINLA ---
            poly_msg = PolygonStamped()
            poly_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Frame ID'nizin TF ağacındaki doğru frame olduğundan emin olun
            poly_msg.header.frame_id = "zed2_left_camera_optical_frame" 
            
            # Numpy array'i int32'ye çevir (Çizim ve mesaj için)
            points = best_mask_data.astype(np.int32)
            

            
            for p in points:
                point_msg = Point32()
                point_msg.x = float(p[0])
                point_msg.y = float(p[1]) 
                point_msg.z = 0.0
                poly_msg.polygon.points.append(point_msg)

            self.poly_pub.publish(poly_msg)
            
            # Görselleştirme: Tüm noktaları çiz (Sarı çizgi)
            cv2.polylines(annotated_frame, [points], True, (0, 255, 255), 2)

        # Sonucu Göster
        cv2.imshow("ZED YOLO Park Tespiti", annotated_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ZedYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()