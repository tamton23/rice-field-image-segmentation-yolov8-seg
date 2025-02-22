from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11n-seg.pt")

# Train the model
results = model.train(data="labelme_json_dir/YOLODataset/dataset.yaml", epochs=50, imgsz=0)
