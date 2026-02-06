from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Export optimized ONNX
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
