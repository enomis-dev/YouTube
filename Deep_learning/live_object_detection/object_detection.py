import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")  # nano model = super fast

cap = cv2.VideoCapture(0)   # your laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    results = model.track(frame, persist=True)

    # Visualize output
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
