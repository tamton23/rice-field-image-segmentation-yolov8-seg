from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("runs/segment/train22/weights/best.pt")  # load a custom model

# Predict with the model
results = model("DJI_20240430091155_0001_D.JPG")  # predict on an image

# Access the results
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
