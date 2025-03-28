# ZeroToNN
从0开始的神经网络训练实现
使用MNIST数据集作为示例

1. 解压原始训练数据。因为原始数据集是复杂的二进制数据，我根本不知道怎么用，因此求助了ai，ai给了我解压脚本：
```shell
cd data
python MnistExtract.py
```

现在所有训练数据都储存在路径`./data/mnist_sorted/` 下


![神经网络](fig_backup/image.png)

这个神经网络的示例中，输入是784(`28*28`)个元素的一个列表，首先要尝试把图片读取，并转换为这样784个0-1数字的列表。

```python
import numpy as np
from PIL import Image

def image_to_flatten_vector(image_path):
    """ 读取图片并转换为 784 维 0-1 之间的数组 """
    img = Image.open(image_path) # 打开图片
    #print(img.mode) # 输出: L，即灰度图 确认是否是灰度图
    #img = img.convert('L')  # 转换为灰度图
    #print(img.size) # 输出: (28, 28)，即 28x28 像素 确认是否是 28x28 像素
    #img = img.resize((28, 28))  # 确保大小为 28x28
    img_array = np.array(img, dtype=np.float16) / 255.0  # 归一化到 0-1 原始的值为 0-255 
    return img_array.flatten()  # 展平为 1D 向量

# 示例：读取 "data/mnist_sorted/0/1.png"
image_path = "data/mnist_sorted/0/1.png"
flatten_vector = image_to_flatten_vector(image_path)
print(flatten_vector.shape)  # 输出: (784,)
```

2. 需要一个随时都能把处理后的列表转换回图片可视化的函数

```python
def vector_to_image(vector, save_path=None):
    """ 将 1D 数组转换回 28x28 图片 """
    img_array = vector.reshape(28, 28)  # 变回 2D 数组
    img_array = (img_array * 255).astype(np.uint8)  # 反归一化回 0-255

    # 用 PIL 创建图片
    img = Image.fromarray(img_array)

    # 显示图片
    #img.show()

    # 可选：保存图片
    if save_path:
        img.save(save_path)

    return img

# 示例：将 1D 向量转换回图片
vector = np.random.rand(784)  # 假设是一个 随机定义的 1D 向量
vector_to_image(vector)
vector_to_image(flatten_vector)

def vector_to_image_raw(vector, save_path=None):
    """ 将 1D 数组转换回 raw 图片 """
    img_array = vector  # 变回 2D 数组
    img_array = (img_array * 255).astype(np.uint8)  # 反归一化回 0-255

    # 用 PIL 创建图片
    img = Image.fromarray(img_array)

    # 显示图片
    #img.show()

    # 可选：保存图片
    if save_path:
        img.save(save_path)

    return img

# 示例：将 1D 向量转换回图片
vector = np.random.rand(784)  # 假设是一个 随机定义的 1D 向量
print(vector.shape)  # 输出: (784,) # 确认是 784 行
vector_to_image_raw(flatten_vector)
```
![手写数字图片转换后的向量可视化](fig_backup/image2.png)

3. 试着计算神经网络第一个隐含层的第一个节点中的值
![神经网络第一个隐含层的第一个节点](fig_backup/image3.png)

```python
def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def neuron_output(weights, bias, inputs):
    """ 计算单个神经元的输出 """
    
    # 计算加权和（点积）并加上偏置 np 库的点乘方法
    weighted_sum = np.dot(weights, inputs) + bias
    
    """
    # 计算加权和（点积）并加上偏置 传统方法 便于理解，但是低效
    weighted_sum = 0
    for w, x in zip(weights, inputs):
        weighted_sum += w * x
    weighted_sum += bias
    """
    # 通过 Sigmoid 激活
    output = sigmoid(weighted_sum)
    
    return output
image_path = "data/mnist_sorted/0/1.png"  # 你的图片路径
inputs = image_to_flatten_vector(image_path)
w = np.random.rand(784)  # 784 维权重向量
b = np.random.rand()  # 随机偏置
r=neuron_output(w,b,inputs)
print(r)
```