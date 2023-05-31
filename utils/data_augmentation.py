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

    for shape in json_data['shapes']:
        # This assert might be overkill but better safe that sorry ...
        assert shape['shape_type'] == "polygon"
        polys.append(Polygon(shape['points'], label=shape['label']))

    img_shape = (json_data['imageHeight'], json_data['imageWidth'], 3)
    polys_oi = PolygonsOnImage(polys, shape=img_shape)
    return(polys_oi)

def convert_json(polygon, image_path):
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
    with open("output.json", "w") as json_file:
        json_file.write(json_data)

my_augmenter = iaa.Sequential([
    iaa.GaussianBlur((0.1, 5)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rotate((-45,45))])

augmented = my_augmenter(image = ex_img, polygons = polys_oi)
[type(x) for x in augmented]
convert_json(augmented[1], "../plane/image_0.jpg")
cv2.imwrite("original_img.png",augmented[0])
# [<class 'numpy.ndarray'>, <class 'imgaug.augmentables.polys.PolygonsOnImage'>]
# So you can make a bunch of augmented image/polygon pairs
augmented_list = [my_augmenter(image = ex_img, polygons = polys_oi) for _ in range(10)]
# Now we just make the overlay for viz purposes
overlaid_images = [polys.draw_on_image(img) for img, polys in augmented_list]
cv2.imwrite("augmented_quokka_polys.png", cv2.hconcat(overlaid_images))