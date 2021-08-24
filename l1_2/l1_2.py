import numpy as np
import numpy.random as rn
import pandas as pd
from sklearn.model_selection import KFold  # 获得k个折数
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import warnings

warnings.filterwarnings('ignore')
# %matplotlib  inline

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# win
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def f_lambda(A, x, y, mu, lambda_0, thresholding_value):
    """
    功能：希望给定xi---> b_mu---->f_lamda_mu
    :param xi:
    :return:
    """
    PI = 3.141493
    n = x.shape[0]
    x_trans = [0] * n  # 初始化计算所得的矩阵为0
    b_mu = x + mu * np.dot(A.T, y - np.dot(A, x))
    for i in range(n):
        if abs(b_mu[i]) > thresholding_value:
            val_i = (lambda_0 / 8) * np.abs(b_mu[i] / 3) ** (-1.5)  # np.power(a, b) ,notice b >0
            if np.abs(val_i) > 1:  # 防止val_i的值大于1，使得arccos报错， 进行最大最小标准化
                val_i = (val_i - thresholding_value) / val_i
            ph_i = np.arccos(val_i)
            x_trans[i] = 2 / 3 * b_mu[i] * (1 + np.cos(2 / 3 * (PI - ph_i)))
    return np.array(x_trans)


# 例子
'''
if __name__ == '__main__':
    n, m = 200, 20
    A = rn.randn(n, m)
    X = rn.randn(m, )
    y = np.dot(A, X) + rn.randn(n, )
    mu = 1 / np.sum(A**2)
    thresholding_value = np.power(54, 1 / 3) / 4 * np.power(lambda_0 * mu, 2 / 3)
    x0 = rn.randn(m, )
    lambda_0 = 0.1
    x = f_lambda(A, x0, y, mu, lambda_0, thresholding_value)
    print(x)
'''


# 损失函数的定义：
def loss_value(A, x, y, lambda_0):
    """损失函数定义"""
    loss = np.sum((y - np.dot(A, x)) ** 2)
    penal = lambda_0 * np.sum(np.abs(x) ** 0.5)  # x0可能有负数
    return loss + penal


# result最小损失值
def loss_function(A, x0, y, mu, lambda_0, max_iter=500, tol=1e-8):
    """
    :param A:
    :param x0: x的初始化
    :param y:
    :param mu: 超参数
    :param lambda_0: 超参数
    :param max_iter: 最大迭代次数
    :param tol: 容忍度
    :return: 训练结果x, 以及损失值
    """
    best_loss = np.inf

    thresholding_value = np.power(54, 1 / 3) / 4 * np.power(lambda_0 * mu, 2 / 3)
    for i in range(max_iter):
        x0 = f_lambda(A, x0, y, mu, lambda_0, thresholding_value)  # 上一步的x0，赋值给x0，便于迭代
        loss = loss_value(A, x0, y, lambda_0)
        if (abs(best_loss - loss) < tol) and (i > 5):
            best_loss = loss
            x_best = x0
            break
        if loss < best_loss:
            best_loss = loss
            x_best = x0          # 记录当前最优损失对应的x0
    return x_best, best_loss


# 例子
'''

if __name__ == '__main__':
    rn.seed(2211)
    n, m = 200, 20
    A = rn.randn(n, m)
    X = np.array([1,2,3,4,5,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    y = np.dot(A, X) + rn.randn(n,)
    x0 = rn.randn(m,)
    mu = 1 / np.sum(A**2)
    lambda_0 = 0.2
    x_best, loss= loss_function(A, x0, y, mu, lambda_0)
    print('x_best: \n{} \n loss\n{}'.format(x_best, loss))
#     x_best = loss_function(A, x0, y, mu, lambda_0)[0]
#     print(x_best)

'''


# 选择最优的lambda_0
def select_best_lambda(A, y, x0, mu):
    """
    :param A:
    :param y:
    :param x:
    :param mu:
    :param max_iter:
    :param tol:
    :return: best_lambda
    """
    # cross valication
    lambda_all = np.arange(1, 100) * 0.01  # 定义lambda的范围与步长
    n = lambda_all.shape[0]
    k = 10  # 进行几次交叉验证
    loss_fun_all = []  # 记录每次的lambda[i],以及每个lambda[i]对应的平均loss_test，
    x_train_all = []
    sk_fold = KFold(n_splits=k, shuffle=True, random_state=111)  # 采用交叉验证函数取出非重合的k折交叉数据将最后一折用于测试
    for i in range(n):
        x_train_i = []
        loss_temp_test = []  # 保存每个lambda_all[i]对应的loss_test
        for train_index, dev_index in sk_fold.split(A, y):  # 依次取出用于训练和测试的索引
            x_train = loss_function(A[train_index], x0, y[train_index], mu=mu, lambda_0=lambda_all[i], max_iter=100, tol=1e-5)[0]  # 只需记录每一次的x_0的值用于计算预测的损失，记录该训练集下对应的训练参数
            loss_ = loss_value(A[dev_index], x_train, y[dev_index], lambda_all[i])  # 用训练处的x_train,计算出对应测试结果
            loss_temp_test.append(loss_)
            x_train_i.append(x_train)
        loss_fun_all.append([lambda_all[i], np.mean(np.array(loss_temp_test))])  # 将lambda[i] 对应的k折损失求平均，作为该lambda值下对应的损失
        x_train_all.append(np.mean(np.array(x_train_i), axis=0))
    loss_fun_all = np.array(loss_fun_all)  # 转换为np.array格式， 便于取出最小的loss对应的索引， 根据该索引求出best_lambda
    #     print(loss_fun_all)
    min_index = np.argmin(loss_fun_all[:, 1])  # 求最小损失对应的索引
    lambda_best = loss_fun_all[min_index][0]  # 取出该行第一个即为最优的best_lambda
    return lambda_best, np.mean(np.array(x_train_all), axis=0)


# 例子
'''
if __name__ == '__main__':
    rn.seed(2211)
    n, m = 200, 20
    A = rn.randn(n, m)
    X = np.array([10, 9, 3,4,5,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    y = np.dot(A, X) + rn.randn(n)
    x0 = rn.randn(m)
    mu = 1 / np.sum(A**2)
    lambda_0 = select_best_lambda(A, y, x0, mu=mu)
    print(lambda_0)
'''

# 函数调用
# 对于数据A, y, 数据要划分成三部分， 第一部分valication,第二部分training, 第三部分testing val:train(train1, test) = 2:8--->val:train:test = 2:7:1
# 先训练出最优的lambda， 再用最优的lambda训练目标函数

if __name__ == "__main__":
    rn.seed(2211)
    n, m = 2000, 20
    A = rn.randn(n, m)
    X = np.array([10, 90, 30, 40, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y = np.dot(A, X) + rn.randn(n)
    x0 = rn.randn(m)
    mu = 1 / np.sum(A ** 2)
    # val
    A_val, A_train, y_val, y_train = train_test_split(A, y, test_size=0.2)  # 划分val，train
    lambda_best, x_train_val = select_best_lambda(A, y, x0, mu=mu)  # 求出最优lambda_best
    print('the best lambda: ', lambda_best)
    # train  = train1 + test
    A_train1, A_test, y_train1, y_test = train_test_split(A_train, y_train, test_size=0.2)
    # 训练模型 x_result
    x_result = loss_function(A_train1, x_train_val, y_train1, mu, lambda_best, max_iter=500, tol=1e-8)[
        0]  # 将val计算出的x,作为初始值。
    print('the trained value of x: \n', x_result)
    # 用训练出的x_result预测y， 计算损失
    loss = loss_value(A_test, x_result, y_test, lambda_best)
    print('the predict loss :\n', loss)
