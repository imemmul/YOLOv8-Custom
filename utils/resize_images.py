import os
from PIL import Image
directory = "/Users/emirulurak/Desktop/dev/datasets/Inovako_dataset/"
classes = os.listdir(directory)

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


if __name__ == "__main__":
    print(test_dataset(640))
            
