import os
import gzip
import numpy as np
from PIL import Image

def load_mnist_images(file_path):
    """ 读取 MNIST 图片数据，确保大端解析 """
    with gzip.open(file_path, 'rb') as f:
        # 读取前 16 个字节（magic, num, rows, cols）
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=">u4")  # 读取大端 4 字节整数
        print(f"Magic: {magic}, Num: {num}, Rows: {rows}, Cols: {cols}")  # 打印调试

        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    
    return images
def load_mnist_labels(file_path):
    """ 读取 MNIST 标签数据 """
    with gzip.open(file_path, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32, count=2)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_images_by_label(images, labels, output_dir):
    """ 将图片按标签分类存储 """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(10):  # 创建 0-9 的文件夹
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(output_dir, str(label), f"{idx}.png")
        Image.fromarray(image).save(image_path)
        
        if idx % 1000 == 0:
            print(f"Saved {idx} images...")

if __name__ == "__main__":
    image_file = "train-images-idx3-ubyte.gz"
    label_file = "train-labels-idx1-ubyte.gz"
    output_directory = "mnist_sorted"
    
    print("Loading images...")
    images = load_mnist_images(image_file)
    print("Loading labels...")
    labels = load_mnist_labels(label_file)
    
    print("Saving images by label...")
    save_images_by_label(images, labels, output_directory)
    
    print("All images saved!")
