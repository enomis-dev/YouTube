"""
Simple YOLOv11n ONNX Inference with Picamera2
"""

from picamera2 import Picamera2
import cv2
import time
import numpy as np
import onnxruntime as ort


class SimpleYOLO:
    def __init__(self, model_path="yolo11n.onnx", conf=0.2, iou=0.5):
        self.conf = conf
        self.iou = iou
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # COCO class names
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def __call__(self, frame):
        # Preprocess: convert RGB to blob
        img = frame.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img})[0]
        
        # Postprocess
        detections = self._postprocess(outputs, frame.shape[:2])
        return detections
    
    def _postprocess(self, output, img_shape):
        # Transpose output [1, 84, 8400] -> [8400, 84]
        predictions = output[0].transpose()
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get class IDs and confidences
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > self.conf
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert boxes from center format to corner format
        x_center, y_center, width, height = boxes.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale to image size (640x640 -> actual size)
        h, w = img_shape
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        
        # NMS
        boxes_nms = np.stack([x1, y1, x2, y2], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(),
            confidences.tolist(),
            self.conf,
            self.iou
        )
        
        # Format results
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'box': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                    'conf': float(confidences[i]),
                    'class': int(class_ids[i]),
                    'name': self.classes[int(class_ids[i])]
                })
        
        return detections
    
    def plot(self, frame, detections):
        """Draw bounding boxes on frame"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['conf']
            name = det['name']
            
            # Color based on class
            np.random.seed(det['class'])
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
        
        return annotated


# Load model
model = SimpleYOLO("yolo11n.onnx", conf=0.2, iou=0.5)

# Setup camera
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (640, 640), "format": "RGB888"}
)
picam2.configure(video_config)
picam2.start()
time.sleep(2)  # camera warm-up

prev_time = time.time()

try:
    while True:
        frame = picam2.capture_array()

        # Run detection
        detections = model(frame)

        # Draw results
        annotated = model.plot(frame, detections)

        # ---- FPS calculation ----
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Put FPS text on frame
        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Show
        cv2.imshow("YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
