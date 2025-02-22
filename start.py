from ultralytics import YOLO
import os
import glob
# Tải mô hình phân đoạn đã huấn luyện
model = YOLO("../runs/detect/train/weights/best.pt")
i = 0
# Dự đoán phân đoạn trên hình ảnh mới
file_path = '../Data_set/'
for filenamepath in glob.glob(os.path.join(file_path, '*.JPG')):
        results = model(filenamepath)
        # Scale image
        i += 1
        print("{}".format(i))
        for result in results:
            result.save(filename="Data_set/"+ filenamepath[12:])
