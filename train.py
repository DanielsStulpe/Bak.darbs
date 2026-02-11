from ultralytics import YOLO

model = YOLO("yolo12m.pt")

model.train(
    data="roboflow_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    workers=1,
    device=0  # GPU (Jetson)
)