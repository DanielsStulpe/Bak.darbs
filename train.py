from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="roboflow_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    device=0  # GPU (Jetson)
)