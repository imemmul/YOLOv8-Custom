# # TODO this executable aims to increase number of data in dataset.
import json
import numpy as np
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import labelme
import base64
dataset_dir = "/home/emir/Desktop/dev/dataset/"

def make_polys(json_file):
    with open(json_file, "r") as js:
        json_data = json.load(js)
    # print(f"json_data {json_data}")
    polys = []
    image_path = json_data['imagePath']
    for shape in json_data['shapes']:
        # This assert might be overkill but better safe that sorry ...
        assert shape['shape_type'] == "polygon"
        polys.append(Polygon(shape['points'], label=shape['label']))

    img_shape = (json_data['imageHeight'], json_data['imageWidth'], 3)
    polys_oi = PolygonsOnImage(polys, shape=img_shape)
    return(polys_oi), image_path

def get_image_data(img_dir):
    data = labelme.LabelFile.load_image_file(img_dir)
    image_data = base64.b64encode(data).decode('utf-8')
    return image_data


def convert_json(polygon, image_path, filename):
    polys = polygon.polygons
    h, w, _ = polygon.shape
    # print(f"image_data {h}x{w}x{_}")
    # print(f"polys {polys}")
    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "imagePath": image_path,
        "imageHeight": h,
        "imageWidth": w,
        "shapes": [],
        "imageData": get_image_data(image_path)
    }
    for pg in polys:
        label = pg.label
        # print(f"pg {pg.exterior} and type of pg {type(pg)}")
        points = pg.exterior
        shape = {
            "label": label,
            "points": points.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)
    json_data = json.dumps(labelme_data, indent=4)
    with open(filename, "w") as json_file:
        json_file.write(json_data)

augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rotate((-45,45)),
    iaa.Crop(percent=(0, 0.01)),
    iaa.Dropout([0.005, 0.01]),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.AddToHueAndSaturation((-60, 60)),  # change their color
    ], random_order=True)

import os
def augmentation(train_dir, val_dir):
    file_count = 120
    for dir in os.listdir(train_dir):
        if dir[-3:] == "jpg":
            img_dir = train_dir + dir
            annot_dir = train_dir + dir[:-3] + "json"
            img = cv2.imread(img_dir)
            polys, dump = make_polys(annot_dir)
            print(f"Converting train {file_count}")
            for _ in range(100): # 50 new augmented images for per images
                augmented_img = augmenter(image=img, polygons=polys)
                cv2.imwrite(f"{train_dir}image_{file_count}.jpg", augmented_img[0])
                convert_json(polygon=augmented_img[1], image_path=f"{train_dir}image_{file_count}.jpg", filename=f"{train_dir}image_{file_count}.json")
                file_count += 1
    for dir in os.listdir(val_dir):
        if dir[-3:] == "jpg":
            img_dir = val_dir + dir
            annot_dir = val_dir + dir[:-3] + "json"
            img = cv2.imread(img_dir)
            polys, dump = make_polys(annot_dir)
            print(f"Converting val {file_count}")
            for _ in range(100): # 50 new augmented images for per images
                augmented_img = augmenter(image=img, polygons=polys)
                cv2.imwrite(f"{val_dir}image_{file_count}.jpg", augmented_img[0])
                convert_json(polygon=augmented_img[1], image_path=f"{val_dir}image_{file_count}.jpg", filename=f"{val_dir}image_{file_count}.json")
                file_count += 1
            
if __name__ == "__main__":
    train_dir = dataset_dir + "train/"
    val_dir = dataset_dir + "val/"
    augmentation(train_dir=train_dir, val_dir=val_dir)