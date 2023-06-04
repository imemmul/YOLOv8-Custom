from ultralytics import YOLO
import os
import cv2

base_weights = "/home/emir/Desktop/dev/dataset_main/yolov8l-seg.pt"
trained_weights = "/home/emir/Desktop/dev/dataset_main/runs/segment/train/weights/best.pt"

dataset = "/home/emir/Desktop/dev/dataset_main/test/"
model = YOLO(base_weights)  # load an official model
model = YOLO(trained_weights)
test_dir = "/home/emir/Desktop/dev/dataset_main/test/"
data_dir = "/home/emir/Desktop/dev/dataset_main/dataset.yaml/"

results = model.val(data=test_dir)
print(f"Results: {results}")


# metrics = model.val(dataset)
# print(metrics)