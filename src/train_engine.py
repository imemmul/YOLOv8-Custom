from ultralytics import YOLO
import os
import cv2


trained_weights = "/home/emir/Desktop/dev/temp_dataset/runs/segment/train/weights/best.pt"
dataset = "/home/emir/Desktop/dev/temp_dataset/dataset.yaml"
model = YOLO(trained_weights)
test_dir = "/home/emir/Desktop/dev/temp_dataset/test/images/"

for dir in os.listdir(test_dir):
    model.predict(test_dir+dir, save=True, conf=0.5, save_txt=True)


# metrics = model.val(dataset)
# print(metrics)