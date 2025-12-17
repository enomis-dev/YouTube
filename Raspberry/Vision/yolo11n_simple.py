from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO
import os

# Name of the YOLO model file
model_path = "yolo11n.pt"

# Automatically download the model if not present
if not os.path.exists(model_path):
    print(f"Model {model_path} not found locally. Downloading...")
    YOLO(model_path)  # Ultralytics will download automatically

# Load YOLO model
model = YOLO(model_path)  # nano model, faster for edge

# Initialize PiCamera2 with smaller frame size
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(preview_config)
picam2.start()
time.sleep(1)  # Let camera warm up

frame_count = 0
skip_frames = 2  # Run YOLO every 2 frames, we don't need it for every frame for a basic application

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Convert RGBA to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Only run YOLO on every `skip_frames` frame
        if frame_count % skip_frames == 0:
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

        frame_count += 1

        # Convert RGB → BGR for OpenCV display on the screen
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLO Object Tracking", annotated_frame_bgr)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
