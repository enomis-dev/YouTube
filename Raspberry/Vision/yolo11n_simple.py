from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (640, 640), "format": "RGB888"}
)
picam2.configure(video_config)
picam2.start()

time.sleep(2)  # camera warm-up

try:
    while True:
        frame = picam2.capture_array()

        # Run detection
        results = model(
            frame,
            conf=0.2,
            iou=0.5,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow("YOLO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
