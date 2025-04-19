from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Nano model (fast but less accurate)

# Train the model
model.train(data="C:/xampp/htdocs/NewProject/dataset/dataset.yaml", epochs=50, batch=16, imgsz=640)
