import os
from PIL import Image
import shutil
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
        print(f"{dest}: annotations size {len(os.listdir(dest))}")
        print(f"For {src}: {len(os.listdir(src))} files left.")
        for annot in os.listdir(src):
            print(f"Moving {src+annot} to {dest+annot}")
            shutil.move(src=src+annot, dst=dest+annot)
        print(f"For {src}: {len(os.listdir(src))} files left.")
        print(f"src {src} len is {len(os.listdir(src))}")
        print(f"src {dest} len is {len(os.listdir(dest))}")

def split_dataset(train_dest, test_dest, split_rate=1/8):
    file_count = 0
    for cls in classes:
        root_dir = directory + cls + "/"
        data_dir = sorted(os.listdir(root_dir))
        test_size = int(len(data_dir) // 2 * split_rate)
        train_size = int(len(data_dir)) // 2 - test_size
        train_data = data_dir[:train_size*2]
        test_data = data_dir[train_size*2:]
        # print(len(train_data))
        # print(len(test_data))
        image_count = 0
        annot_count = 0
        for dir in test_data:
            if dir[-3:] == "jpg":
                image_count += 1
            else:
                annot_count += 1
        print(f"image_count of train {image_count}")
        print(f"annot_count of train {annot_count}")
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
    # print(f"file count is {file_count}")
    assert file_count * 2 == len(os.listdir(train_dest)) + len(os.listdir(test_dest))
            

if __name__ == "__main__":
    # print(test_dataset(640))
    # move_annotat"ions()
    split_dataset(directory+"train/", directory+"test/")
    # move_annotations()
