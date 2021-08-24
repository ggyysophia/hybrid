import numpy as np
from numpy import genfromtxt
from sklearn import linear_model

# 读取数据
path = r'/Users/snszz/PycharmProjects/hybrid_sl_code_python/资料/程序/回归/longley.csv'
data = genfromtxt(path , delimiter=',')

# 切分数据
x_data = data[1:, 2:]
y_data = data[1:, 1, np.newaxis]

# 训练模型
model = linear_model.ElasticNetCV()
model.fit(x_data, y_data)

# 弹性网系数
print(model.alpha_)
# 回归模型参数
print(model.coef_)

# 预测值
print(model.predict(x_data[-2, np.newaxis]))
print(y_data[-2])  # 真实值

