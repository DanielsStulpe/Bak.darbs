from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    device="cpu"  # GPU (Jetson)
)