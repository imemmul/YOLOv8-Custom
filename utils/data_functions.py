import os
from PIL import Image
import shutil
import labelme
import base64
directory = "/home/emir/Desktop/dev/Inovako_dataset_fully/"

classes = ["plane", "elephant", "dinosaur"]

def resize_images(size):
    for cls in classes:
        class_dir = directory + cls + '/'
        for img_dir in os.listdir(class_dir):
            img = Image.open(class_dir+img_dir)
            print(f"Changing size to {size}x{size} of {cls}:{img_dir}")
            img = img.resize((size, size))
            img.save(class_dir+img_dir)

def test_dataset(size):
    for cls in classes:
        if cls != ".DS_Store":
            class_dir = directory + cls + '/'
            annot_dir = directory + f"{cls}_annotations/"
            print(f"class_dir {class_dir}")
            print(f"Number of instances of class: {cls}: {len(os.listdir(class_dir))}")
            assert len(os.listdir(class_dir)) in [40,80]
            assert len(os.listdir(annot_dir)) in [40, 0]
            print(f"Number of instances of class annotations: {cls}: {len(os.listdir(annot_dir))}")
    return True

def test_train_test(train_dir, test_dir, val_dir):
    for f in os.listdir(train_dir): 
        if f[-3:] == "jpg":
            dir = train_dir + f[:-3] + "json"
            if not os.path.isfile(dir):
                print(dir)
                return False
    for f in os.listdir(test_dir): 
        if f[-3:] == "jpg":
            dir = test_dir + f[:-3] + "json"
            if not os.path.isfile(dir):
                print(dir)
                return False
    for f in os.listdir(val_dir): 
        if f[-3:] == "jpg":
            dir = val_dir + f[:-3] + "json"
            if not os.path.isfile(dir):
                print(dir)
                return False
    return True
            
def move_annotations():
    """
    to convert YOLOv8 format annotations of labelme and the source images should be in the same folder
    """
    for cls in classes:
        src = f"{directory}{cls}_annotations/"
        dest = f"{directory}{cls}/"
        print(f"{dest}: annotations size {len(os.listdir(dest))}")
        print(f"For {src}: {len(os.listdir(src))} files left.")
        for annot in os.listdir(src):
            print(f"Moving {src+annot} to {dest+annot}")
            shutil.move(src=src+annot, dst=dest+annot)
        print(f"For {src}: {len(os.listdir(src))} files left.")
        print(f"src {src} len is {len(os.listdir(src))}")
        print(f"src {dest} len is {len(os.listdir(dest))}")


def check_size(data):
    image_count = 0
    annot_count = 0
    for dir in data:
        if dir[-3:] == "jpg":
            image_count += 1
        else:
            annot_count += 1
    print(f"image_count of train {image_count}")
    print(f"annot_count of train {annot_count}")


def split_dataset(train_dest, test_dest, val_dest,):
    file_count = 0
    for cls in classes:
        root_dir = directory + cls + "/"
        data_dir = sorted(os.listdir(root_dir))
        test_size = 10
        val_size = 10
        train_size = 60
        train_data = data_dir[:train_size]
        test_data = data_dir[train_size:test_size+train_size]
        val_data = data_dir[train_size+test_size:]
        # print(len(train_data))
        # print(len(test_data))
        print(f"train data:")
        check_size(train_data)
        print(f"test data:")
        check_size(test_data)
        print(f"val data:")
        check_size(val_data)
        
        for dir in train_data:
            if dir[-3:] == "jpg":
                move_img_dir = root_dir + dir
                move_annot_dir = root_dir + dir[:-3] + "json"
                dest_img_dir = train_dest + f"image_{file_count}.jpg"
                dest_annot_dir = train_dest + f"image_{file_count}.json"
                print(f"moved {move_img_dir} to dest {dest_img_dir}")
                print(f"moved {move_annot_dir} to dest {dest_annot_dir}")
                shutil.move(move_img_dir, dest_img_dir)
                shutil.move(move_annot_dir, dest_annot_dir)
                file_count += 1
        for dir in test_data:
            if dir[-3:] == "jpg":
                move_img_dir = root_dir + dir
                move_annot_dir = root_dir + dir[:-3] + "json"
                dest_img_dir = test_dest + f"image_{file_count}.jpg"
                dest_annot_dir = test_dest + f"image_{file_count}.json"
                # print(f"image: {move_img_dir}")
                # print(f"annot: {move_annot_dir}")
                print(f"moved {move_img_dir} to dest {dest_img_dir}")
                print(f"moved {move_annot_dir} to dest {dest_annot_dir}")
                shutil.move(move_img_dir, dest_img_dir)
                shutil.move(move_annot_dir, dest_annot_dir)
                file_count += 1
        for dir in val_data:
            if dir[-3:] == "jpg":
                move_img_dir = root_dir + dir
                move_annot_dir = root_dir + dir[:-3] + "json"
                dest_img_dir = val_dest + f"image_{file_count}.jpg"
                dest_annot_dir = val_dest + f"image_{file_count}.json"
                # print(f"image: {move_img_dir}")
                # print(f"annot: {move_annot_dir}")
                print(f"moved {move_img_dir} to dest {dest_img_dir}")
                print(f"moved {move_annot_dir} to dest {dest_annot_dir}")
                shutil.move(move_img_dir, dest_img_dir)
                shutil.move(move_annot_dir, dest_annot_dir)
                file_count += 1
    print(f"file count is {file_count}")
    # assert file_count * 2 == (len(os.listdir(train_dest)) + len(os.listdir(test_dest) + len(os.listdir(val_dest))))
    
import json

def fix_image_paths(fix_dir):
    for dir in os.listdir(fix_dir):
        if dir[-4:] == "json":
            json_file = fix_dir + dir
            print(f"json file to read {json_file}")
            with open(json_file, "r") as js:
                json_data = json.load(js)
            # print(f"json_data {json_data}")
            print(f"json_file {json_file}")
            print(f"jason_data before {json_data['imagePath']}")
            json_data['imagePath'] = "./" + dir[:-4] + "jpg"
            print(f"jason_data after {json_data['imagePath']}")
            with open(json_file, "w") as j_file:
                json.dump(json_data, j_file, indent=4)
                
import numpy as np
def check_pixel_values(dir):
    img = Image.open(dir)

    # Convert the image data to a numpy array.
    data = np.array(img)

    # Check if there are any negative values in the array.
    if (data < 0).any():
        print("The image has negative pixel values.")
    else:
        print("The image does not have negative pixel values.")
def fix_image_data(fix_dir):
    for dir in os.listdir(fix_dir):
        if dir[-4:] == "json":
            json_file = fix_dir + dir
            img_path = fix_dir + dir[:-4] + "jpg"
            print(f"what is img_path {img_path}")
            print(f"json file to read {json_file}")
            with open(json_file, "r") as js:
                json_data = json.load(js)
            # print(f"json_data {json_data}")
            print(f"json_file {json_file}")
            print(f"jason_data before {json_data['imageData']}")
            data = labelme.LabelFile.load_image_file(img_path)
            image_data = base64.b64encode(data).decode('utf-8')
            json_data['imageData'] = image_data
            print(f"jason_data after {json_data['imageData']}")
            with open(json_file, "w") as j_file:
                json.dump(json_data, j_file, indent=4)




if __name__ == "__main__":
    # print(test_dataset(640))
    # move_annotations()
    # split_dataset(directory+"train/", directory+"test/")
    train_dir = "/home/emir/Desktop/dev/dataset/train/" # 90 images 30 per class
    test_dir = "/home/emir/Desktop/dev/dataset/test/" # 15 images 5 per class
    val_dir = "/home/emir/Desktop/dev/dataset/val/" # 15 images 5 per class
    dir = "/home/emir/Desktop/dev/dataset_main/"
    # split_dataset(train_dest=train_dir, test_dest=test_dir, val_dest=val_dir)
    # print(test_train_test(train_dir=train_dir, test_dir=test_dir, val_dir=val_dir))
    # move_annotations()
    # split_dataset(train_dest=train_dir, test_dest=test_dir, val_dest=val_dir)
    # print(test_train_test(train_dir=dir+"train/", test_dir=test_dir, val_dir=val_dir))
    print(len(os.listdir(dir+"train/images/")))
    print(len(os.listdir(dir+"val/images/")))
    print(len(os.listdir(dir+"test/images/")))
    
    print(len(os.listdir(dir+"train/labels/")))
    print(len(os.listdir(dir+"val/labels/")))
    print(len(os.listdir(dir+"test/labels/")))