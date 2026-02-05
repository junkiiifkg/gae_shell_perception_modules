import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from gae_msgs.msg import GaeShellPolygon

Debug = True  # Debug görselleştirme için (center point görmek için True yapıldı)

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
        
        self.debug_pub = self.create_publisher(Image, "/yolo/annotated", 10)
        
        # 4. Publisher (Park Alanı Poligonu)
        self.poly_pub = self.create_publisher(GaeShellPolygon, '/parking_area/polygon', 10)
        
        self.get_logger().info("YOLO Detector Hazır. Topic: /parking_area/polygon")
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge Hatası: {e}")
            return
        
        # Tahmin Yap
        results = self.model(frame, verbose=False)
        result = results[0]
        annotated_frame = frame.copy()
        
        # Maske ve güven skoru kontrolü
        if result.masks is not None and len(result.boxes.conf) > 0:
            # --- EN İYİ TESPİTİ SEÇME ---
            scores = result.boxes.conf
            best_index = torch.argmax(scores).item()
            
            # En iyi maske verisini al (Numpy array: [N, 2])
            best_mask_data = result.masks.xy[best_index]
            
            # --- POLIGON MESAJINI OLUŞTUR VE YAYINLA ---
            GaeShellPolygon_msg = GaeShellPolygon()
            GaeShellPolygon_msg.header.stamp = self.get_clock().now().to_msg()
            GaeShellPolygon_msg.header.frame_id = "zed2_left_camera_optical_frame"
            GaeShellPolygon_msg.confidence = float(scores[best_index].item())
            
            if GaeShellPolygon_msg.confidence < 0.75:
                return  # Güven skoru düşükse yayınlama
            
            poly_msg = PolygonStamped()
            poly_msg.header.stamp = self.get_clock().now().to_msg()
            poly_msg.header.frame_id = "zed2_left_camera_optical_frame" 
            
            # Numpy array'i int32'ye çevir (Çizim ve mesaj için)
            points = best_mask_data.astype(np.int32)
            
            for p in points:
                point_msg = Point32()
                point_msg.x = float(p[0])
                point_msg.y = float(p[1]) 
                point_msg.z = 0.0
                poly_msg.polygon.points.append(point_msg)
            
            GaeShellPolygon_msg.polygon = poly_msg
            self.poly_pub.publish(GaeShellPolygon_msg)
            
            # === VİZUALİZASYON: POLİGON + MERKEZ NOKTA ===
            
            # 1. Poligon çizimi (Sarı çizgi)
            cv2.polylines(annotated_frame, [points], True, (0, 255, 255), 2)
            
            # 2. Merkez noktası hesapla
            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))
            
            # 3. Merkez noktası çiz (Kırmızı daire + çarpı işareti)
            cv2.circle(annotated_frame, (center_x, center_y), 8, (0, 0, 255), -1)  # Dolu kırmızı daire
            cv2.circle(annotated_frame, (center_x, center_y), 12, (0, 0, 255), 2)   # Dış çember
            
            # Çarpı işareti
            cv2.line(annotated_frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 255), 2)
            cv2.line(annotated_frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 255), 2)
            
            # 4. Koordinat bilgisi yazdır
            text = f"Center: ({center_x}, {center_y})"
            cv2.putText(annotated_frame, text, (center_x + 20, center_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 5. Güven skoru yazdır
            conf_text = f"Confidence: {GaeShellPolygon_msg.confidence:.2f}"
            cv2.putText(annotated_frame, conf_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 6. Polygon köşe sayısı
            corner_text = f"Points: {len(points)}"
            cv2.putText(annotated_frame, corner_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Debug görüntüsünü yayınla
        if Debug:
            debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

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