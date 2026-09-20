import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

from ultralytics import YOLO


class AMRDetector(Node):
    def __init__(self):
        super().__init__("amr_detector")

        self.declare_parameter(
            "model_path",
            "/home/deepk12341234/amr_ai/runs/detect/runs/yolo26s_torwic_960/weights/best.pt",
        )
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("detection_topic", "/amr/detections")
        self.declare_parameter("imgsz", 960)
        self.declare_parameter("device", "0")

        model_path = self.get_parameter("model_path").value
        image_topic = self.get_parameter("image_topic").value
        detection_topic = self.get_parameter("detection_topic").value
        imgsz = int(self.get_parameter("imgsz").value)
        device = self.get_parameter("device").value

        self.class_names = {
            0: "person",
            1: "fork_truck",
            2: "cart_pallet_jack",
            3: "rack_shelf",
            4: "misc_dynamic_feature",
        }

        self.class_thresholds = {
            0: 0.30,
            1: 0.25,
            2: 0.35,
            3: 0.30,
            4: 0.35,
        }

        self.imgsz = imgsz
        self.device = device

        self.get_logger().info(f"Loading YOLO model: {model_path}")

        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            detection_topic,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.frame_count = 0

        self.get_logger().info("AMR YOLO detector started.")
        self.get_logger().info(f"Image topic: {image_topic}")
        self.get_logger().info(f"Detection topic: {detection_topic}")
        self.get_logger().info(f"Image size: {imgsz}")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=0.001,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return

        result = results[0]

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        if result.boxes is None or len(result.boxes) == 0:
            self.detection_pub.publish(detections_msg)
            return

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, confidence in zip(boxes, classes, confidences):
            cls = int(cls)
            confidence = float(confidence)

            threshold = self.class_thresholds.get(cls, 0.25)

            if confidence < threshold:
                continue

            x1, y1, x2, y2 = box

            detection = Detection2D()
            detection.header = msg.header

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self.class_names.get(
                cls,
                str(cls),
            )
            hypothesis.hypothesis.score = confidence

            detection.results.append(hypothesis)

            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            detection.bbox.center.position.x = float((x1 + x2) / 2.0)
            detection.bbox.center.position.y = float((y1 + y2) / 2.0)
            detection.bbox.size_x = float(width)
            detection.bbox.size_y = float(height)

            detections_msg.detections.append(detection)

        self.detection_pub.publish(detections_msg)

        self.frame_count += 1

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Processed {self.frame_count} frames"
            )


def main(args=None):
    rclpy.init(args=args)

    node = AMRDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
