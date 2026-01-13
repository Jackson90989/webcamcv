from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Users\JacksonRodrigues\Downloads\Output\Find peace signs 3.v3-roboflow-instant-2--eval-.yolov8 (1)\data.yaml",
    epochs=60,
    imgsz=640
)
