from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    device="cpu"  # GPU (Jetson)
)

metrics = model.val()

# Perform object detection on an image 
results = model("https://www.pavementinteractive.org/wp-content/uploads/2007/08/Kailua_pothole.jpg") 
results[0].show()