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


if __name__ == "__main__":
    resize_images(640)
            
