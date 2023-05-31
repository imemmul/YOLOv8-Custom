import os
from PIL import Image
import shutil
directory = "/home/emir/Desktop/dev/Inovako_dataset_fully/"
classes = ["plane", "dinosaur", "elephant"]

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
                print(f"class_dir {class_dir}")
                print(f"Number of instances of class: {cls}: {len(os.listdir(class_dir))}")
                for img_dir in os.listdir(class_dir):
                    ori_dir = class_dir + img_dir
                    image = Image.open(ori_dir)
                    image_format = image.format
                    image_size = image.size
                    if image_size != (size, size):
                        print(f"wrong image size in {img_dir}")
                        return False
                    # print("Image Format:", image_format, "Image Size:", image_size)
                    return True
def move_annotations():
    """
    to convert YOLOv8 format annotations of labelme and the source images should be in the same folder
    """
    for cls in classes:
        src = f"{directory}{cls}_annotations/"
        dest = f"{directory}{cls}/"
        # print(f"{dest}: annotations size {len(os.listdir(dest))}")
        # print(f"For {src}: {len(os.listdir(src))} files left.")
        # for annot in os.listdir(src):
        #     print(f"Moving {src+annot} to {dest+annot}")
        #     shutil.move(src=src+annot, dst=dest+annot)
        # print(f"For {src}: {len(os.listdir(src))} files left.")
        print(f"src {src} len is {len(os.listdir(src))}")
        print(f"src {dest} len is {len(os.listdir(dest))}")
if __name__ == "__main__":
    # print(test_dataset(640))
    move_annotations()
