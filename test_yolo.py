from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val(
    data="roboflow_dataset_yolo/data.yaml",
    split="val",
    imgsz=640,
    conf=0.001,
    iou=0.6,
    device="cpu"
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)

results = model("Kailua_pothole.jpg", conf=0.25)
results[0].show()