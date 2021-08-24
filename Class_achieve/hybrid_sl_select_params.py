"""
% Function to perform parameter selection for hybrid low-rank + sparse
% subspace learning. Chooses k and lambda that minimize AIC score.
%
% Inputs:
%   X: observed data, n-by-p matrix of observed features
%   kVals: vector of candidate values for k
%   lambdaVals: vector of candidate values for lambda (optional)
%   selOption: struct of parameter selection options (optional)
%   optOption: struct of optimization parameters (optional)
%
% Outputs:
%   Z: n-by-k matrix of low-rank component features
%   A: k-by-p matrix of low-rank component coefficients
%   W: n-by-p matrix of high-dim component features
%   b: p-by-1 vector of high-dim component coefficients
%   k: selected value of k
%   lambda: selected value of lambda
%   results: results struct containing loss value, sparsity,
%            and aic score for each parameter setting


"""

import pandas as pd
import numpy as np
from hybrid_sl_wrapper_exclusive import hybrid_sl_wrapper_exclusive
from hybrid_sl_wrapper_overlapping import hybrid_sl_wrapper_overlapping
from Hybrid_SL_REFIT import Hybrid_SL_REFIT


def hybrid_sl_select_params(X, kVals, lambdaVals=[], gamma='max', verbose=0, save=0):
    #  设置默认的lambda_值， matlab与python一样，默认sum，和max都是按列（按列求最大值，按列求和）进行操作
    if lambdaVals == []:
        lambdaMax = max(sum(np.abs(X)))
        lambdaVals = lambdaMax * np.array([2 ** i for i in range(-15, 1)])

    #  损失函数、稀疏性，AIC的初始化存储
    lossVal = np.zeros((len(kVals), len(lambdaVals)))
    lrFrac = np.zeros((len(kVals), len(lambdaVals)))
    hdFrac = np.zeros((len(kVals), len(lambdaVals)))
    aicScore = np.zeros((len(kVals), len(lambdaVals)))

    #  初始最优变,aic[]
    bestAic = np.inf
    bestZ = []
    bestA = []
    bestW = []
    bestb = []

    #  估计每一个k和lambda_
    for kInd in range(len(kVals)):
        k = kVals[kInd]
        for lambdaInd in range(len(lambdaVals)):
            lambda_ = lambdaVals[lambdaInd]

            #  打印状态

            if verbose:
                print('Running with k = {}, lambda = {}...\n'.format(k, lambda_))

            #  选择最大的lambda_, 用数据拟合模型
            if gamma == 'max':
                Z, A, W, b, objVals, gammaVals, aicVals = hybrid_sl_wrapper_exclusive(X, k, lambda_)

            # 选择最优的lambda, 用数据拟合模型
            elif gamma == 'best':
                Z, A, W, b, objVals, gammaVals, aicVals = hybrid_sl_wrapper_overlapping(X, k, lambda_)

            # 计算损失函数
            lossVal[kInd, lambdaInd] = (np.sum(X - np.dot(Z, A) - np.dot(W, np.diag(b))) ** 2) / 2
            # 计算低阶/高阶比例
            lrFrac[kInd, lambdaInd] = np.mean(sum(np.abs(A)) != 0)
            hdFrac[kInd, lambdaInd] = np.mean(np.abs(b) != 0)
            # 计算AIC值
            p1 = sum(sum(np.abs(A)) != 0)
            p2 = sum(np.abs(b) != 0)
            numParam = k * (k - 1) / 2 + k * p1 + 1 + p2 * (p2 - 1) / 2 + p2
            aicScore[kInd, lambdaInd] = 2 * numParam + 2 * lossVal[kInd, lambdaInd]
            # 打印状态
            if verbose:
                print('Result: loss = %g, aic = %g, low-rank/sparse = %.2f/%.2f\n' \
                      % (lossVal[kInd, lambdaInd], aicScore[kInd, lambdaInd], lrFrac[kInd, lambdaInd],
                         hdFrac[kInd, lambdaInd]))

            #  存储结果
            if aicScore[kInd, lambdaInd] <= bestAic:
                bestZ = Z
                bestA = A
                bestW = W
                bestb = b
                bestAic = aicScore[kInd, lambdaInd]

            # refit Z and W
            # Z, W, objValsZW = C.hybrid_sl_refit_(X, A, b, k, initZ=[], initW=[])
            Z, W, objValsZW = Hybrid_SL_REFIT().hybrid_sl_refit_(X, A, b, k, Z, W)
            #  保存结果
            if save:
                np.savez('results_summary', k=k, lambda_=round(lambda_ * 10) / 10, A=A, b=b, Z=Z, W=W, objVals=objVals,
                         gammaVals=gammaVals, aicVals=aicVals)
    # 选择最优的参数
    kInd, lambdaInd = np.where(aicScore == np.min(aicScore))
    k = kVals[kInd[0]]
    lambda_ = lambdaVals[lambdaInd[0]]

    # 打印出最优参数
    if verbose:
        print('Best k = %d, Best lambda = %g\n' % (k, lambda_))

    # 返回最优变量
    Z = bestZ
    A = bestA
    W = bestW
    b = bestb

    # 保存完整的结果
    if save:
        np.savez('results_summary', lossVal=lossVal, lrFrac=lrFrac, hdFrac=hdFrac, aicScore=aicScore)

    #  返回结果
    results = {'lossVal': lossVal, 'lrFrac': lrFrac, 'hdFrac': hdFrac, 'aicScore': aicScore}

    return Z, A, W, b, k, lambda_, results


if __name__ == '__main__':
    np.random.seed(222)
    X = np.random.rand(100, 200)
    kVals = [10, 20]
    Z, A, W, b, k, lambda_, results = hybrid_sl_select_params(X, kVals, lambdaVals=[], gamma='max', verbose=0, save=0)
    print(lambda_)
    print(results)
