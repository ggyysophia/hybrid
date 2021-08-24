# 通过最大化混合的低秩子空间来选择最优的lambda
"""
% Function to optimize the objective function for mutually exclusive
% hybrid low-rank + sparse subspace learning using alternating accelerated
% proximal gradient descent. Specifically, the problem is given by
%
%   min_{Z,A,W,b} sum (1/2) * ||X - Z*A - W*diag(b)||_F^2
%       + C * ||A*diag(b)||_{1,2} + lambda * ||b||_1
%   s.t. ||Z||_F <= 1, ||W||_F <= 1
%
% where C is a constant large enough that ||A*diag(b)||_{1,2} = 0
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

def hybrid_sl_wrapper_oracle(X, k, L_true, innertol=1e-5, outertol=1e-3):
    # 混合子空间的调优网格搜索

    # 混合子空间网格搜索

    # 最优误差的初始化
    bestErr = np.inf

    # 建立候选的lambda值
    lambdaMax = max(sum(np.abs(X)))
    lambdaVals = lambdaMax * np.array(
        [0] + list(np.power(1 / 2, [i for i in range(10, -2, -2)])))

    # 初始化误差
    errSfull = np.zeros(len(lambdaVals))
    errLsub = np.zeros(len(lambdaVals))

    #  non-overlapping case， Ab = 0,一个特征只属于一子集（低秩，）
    if sum(sum(np.abs(L_true)) * sum(abs(S_true))) == 0:

        # 对lambda的多个值运行混和子空间计算
        for i in range(len(lambdaVals)):
            lambda_ = lambdaVals[i]
            Z, A, W, b = hybrid_sl_wrapper_exclusive(X, k, lambda_)
            err = calc_subspace_err(L_true, Z * A, k)
            errLsub[i] = err
            errSfull[i] = calc_recovery_err(S_true, np.dot(
                W, np.diag(b)))  # W * b = np.dot(W,np.diag(b)
            if err < bestErr:
                bestErr = err
                bestZ = Z
                bestA = A
                bestW = W
                bestb = b

        # 重叠的情况
    else:
        for i in range(len(lambdaVals)):
            lambda_ = lambdaVals[i]
            gamma = 0
            Z = []
            A = []
            W = []
            b = []

            while True:
                Z, A, W, b = Hybrid_SL_ACCE().hybrid_sl_optimization_accel(
                    X, k, gamma, 0, lambda_, Z, A, W, b)
                err = calc_subspace_err(L_true, np.dot(Z, A), k)
                if err < bestErr:
                    bestErr = err
                    bestZ = Z
                    bestA = A
                    bestW = W
                    bestb = b

                if sum(sum(np.abs(A)) * np.abs(b.T)) == 0:
                    break
                elif gamma == 0:
                    gamma = 0.25
                else:
                    gamma = 2 * gamma

    return bestZ, bestA, bestW, bestb, bestErr