# 优化重叠的目标函数
"""
% Function to optimize the objective function for overlapping hybrid
% low-rank + sparse subspace learning using alternating accelerated
% proximal gradient descent. Specifically, the problem is given by
%
%   min_{Z,A,W,b} sum (1/2) * ||X - Z*A - W*diag(b)||_F^2
%       + gamma * ||A*diag(b)||_{1,2} + lambda * ||b||_1
%   s.t. ||Z||_F <= 1, ||W||_F <= 1
%
% where gamma is a parameter selected to minimize the AIC score.
%
% Inputs:
%   X: n-by-p matrix of observed features
%   k: dimensionality of low-rank latent space
%   lambda: regularization parameter for individual penalty on b
%   option: struct of optimization parameters (optional)
%
% Outputs:
%   Z: n-by-k matrix of low-rank component features
%   A: k-by-p matrix of low-rank component coefficients
%   W: n-by-p matrix of high-dim component features
%   b: p-by-1 vector of high-dim component coefficients
%   objVals: objective value at each iteration
%   gammaVals: value of gamma at each iteration
%   aicVals: aic score for each value of gamma

"""
import numpy as np
from Hybrid_SL_ACCE import Hybrid_SL_ACCE


def hybrid_sl_wrapper_overlapping(X, k, lambda_, verbose=0):
    # 初始化gamma和变量
    gamma = 0
    Z = []
    A = []
    W = []
    b = []

    # 初始化存储的目标值
    objVals = []
    gammaVals = []

    # 初始化存储AIC值
    aicVals = []

    # 初始化最优变量
    bestAic = np.inf
    bestZ = Z
    bestA = A
    bestW = W
    bestb = b

    # 使用递增的gamma值学习一系列模型
    while True:

        # 使用之前的解初始化，用当前的gamma值进行优化
        Z, A, W, b, currObjVals, objIds = Hybrid_SL_ACCE().hybrid_sl_optimization_accel(X, k, gamma, 0, lambda_, Z, A,
                                                                                        W, b)
        objVals.append(currObjVals)
        gammaVals.append(gamma * np.ones(len(currObjVals)))

        # 计算A,b的密度

        indOnA = sum(abs(A)) > 0
        indOnB = abs(b.T) > 0
        dnsA = 100 * np.mean(indOnA)
        dnsB = 100 * np.mean(indOnB)
        dnsBoth = 100 * np.mean(indOnA & indOnB)
        dnsEither = 100 * np.mean(indOnA | indOnB)

        # 计算AIC值
        p1 = sum(sum(np.abs(A)) != 0)
        p2 = sum(np.abs(b) != 0)
        numParam = k * (k - 1) / 2 + k * p1 + 1 + p2 * (p2 - 1) / 2 + p2
        lossVal = np.sum((X - np.dot(Z, A) - W * b.T) ** 2) / 2
        aicVals.append(2 * numParam + 2 * lossVal)

        if aicVals[-1] <= bestAic:
            bestZ = Z
            bestA = A
            bestW = W
            bestb = b
            bestAic = aicVals[-1]

        if verbose:
            # print("%.2f%%" %25)    #后边两个百分号是输出一个百分号的意思
            # print('{:.2f}%'.format(25))
            # print('{:.2%}'.format(0.25))
            print(
                'Gamma = {}: Dens of A = {:.1f}%, Dens of b = {}%, Overlap = {.1f}%, Coverage = %{:.1f}%\n'
                    .format(gamma, dnsA, dnsB, dnsBoth, dnsEither))

        # 确定停止准则
        if np.sum(dnsBoth) == 0:
            break

        # 更新gamma
        if gamma == 0:
            gamma = .25
        else:
            gamma = 2 * gamma

    # 返回最优变量
    Z = bestZ
    A = bestA
    W = bestW
    b = bestb

    return Z, A, W, b, objVals, gammaVals, aicVals


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    np.random.seed(2222)
    X = np.random.randn(100, 200)
    Z, A, W, b, objVals, gammaVals, aicVals = hybrid_sl_wrapper_overlapping(X=np.random.randn(100, 200), k=20,
                                                                            lambda_=0.2)
    print(objVals)
    print(aicVals)
