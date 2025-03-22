# ZeroToNN
从0开始的神经网络训练实现
使用MNIST数据集作为示例

1. 解压原始训练数据。因为原始数据集是复杂的二进制数据，我根本不知道怎么用，因此求助了ai，ai给了我解压脚本：
```shell
cd data
python MnistExtract.py
```

现在所有训练数据都储存在路径`./data/mnist_sorted/` 下
