# # TODO this executable aims to increase number of data in dataset.
import json
import numpy as np
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

dataset_dir = "/home/emir/Desktop/dev/Inovako_dataset_fully/"

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

def convert_json(polygon, image_path, filename):
    polys = polygon.polygons
    h, w, _ = polygon.shape
    print(f"image_data {h}x{w}x{_}")
    print(f"polys {polys}")
    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "imagePath": image_path,
        "imageHeight": h,
        "imageWidth": w,
        "shapes": []
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

augmenter = iaa.Sequential([ # TODO this is going to be changed
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rotate((-45,45)),
    iaa.Crop(percent=(0, 0.1)),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )])

# [type(x) for x in augmented]
# convert_json(augmented[1], image_path)
# cv2.imwrite("original_img.jpg", ex_img)
# cv2.imwrite("augmented_img.jpg", augmented[0])
# output_segmentation = augmented[1].draw_on_image(augmented[0])
# print(augmented[1])
# cv2.imwrite("augmented_output.jpg", output_segmentation)
# created_polys, new_img_path = make_polys("./output.json")
# new_out = created_polys.draw_on_image(augmented[0])
# cv2.imwrite("new_created_json.jpg", new_out)
# print(f"image paths of original: {image_path} new_image_path: {new_img_path}")

# [<class 'numpy.ndarray'>, <class 'imgaug.augmentables.polys.PolygonsOnImage'>]
# So you can make a bunch of augmented image/polygon pairs
# augmented_list = [augmenter(image = ex_img, polygons = polys) for _ in range(20)]
# # Now we just make the overlay for viz purposes
# concat_augmented = [polys.draw_on_image(img) for img, polys in augmented_list]
# cv2.imwrite("concat_new.png", cv2.hconcat(concat_augmented))
import os
def augmentation():
    train_dir = dataset_dir + "train/"
    file_count = 120
    for dir in os.listdir(train_dir):
        if dir[-3:] == "jpg":
            img_dir = train_dir + dir
            annot_dir = train_dir + dir[:-3] + "json"
            img = cv2.imread(img_dir)
            polys, dump = make_polys(annot_dir)
            for _ in range(20): # 20 new augmented images for per images
                augmented_img = augmenter(image=img, polygons=polys)
                cv2.imwrite(f"{train_dir}image_{file_count}.jpg", augmented_img[0])
                convert_json(polygon=augmented_img[1], image_path=dir, filename=f"{train_dir}image_{file_count}.json")
                file_count += 1
            
if __name__ == "__main__":
    # augmentation()
    train_dir = dataset_dir + "train/"
    print(len(os.listdir(train_dir)))