# 本数据的学习速率选择0.0001，迭代次数为50次，初始参数值为0
import numpy as np
from sklearn import datasets, linear_model
from numpy import genfromtxt
import matplotlib.pyplot as plt

# load data
#加载数据集，是一个字典类似Java中的map
lris_df = datasets.load_iris()
# print('lris_df:', lris_df)
# .data,  150 * 4
# .target, 150 * 1
# target_names  array(['setosa', 'versicolor', 'virginica']
# print(type(lris_df))
# print('data.shape:',lris_df.data.shape)

# 切分数据
x_train, x_test = lris_df.data[:120, :], lris_df.data[120:, :]
y_train, y_test = lris_df.target[:120], lris_df.target[120:]

x_test

# 训练数据
model = linear_model.ElasticNetCV()
model.fit(x_train, y_train)

# 弹性网系数
print(model.alpha_)
# 回归模型系数
print(model.coef_)

# 预测值
print(model.predict(x_test))
print(y_test) # 真实值













# newaxis
'''
对于[: , np.newaxis] 和 [np.newaxis，：]
是在np.newaxis这里增加1维。
这样改变维度的作用往往是将一维的数据转变成一个矩阵，与代码后面的权重矩阵进行相乘， 否则单单的数据是不能呢这样相乘的哦。
'''
# np.genfromtxt(r.'lingery.csv', delimiter = ',')   https://blog.csdn.net/dian19881021/article/details/102195163

