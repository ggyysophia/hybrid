# 使用类实现函数的功能
"""
% Function to refit a trained hybrid low-rank + sparse subspace learning
% model to test data using projected gradient descent. Specifically, the
% problem is given by
%
%   min_{Z,W} (1/2) * ||X - Z*A - W*diag(b)||_F^2
%   s.t. ||Z||_F <= 1, ||W||_F <= 1 for i = 1,...,n
%
% where A and b were previously estimated on a training set
%
% Inputs:
%   X: n-by-p matrix of observed features
%   A: k-by-p matrix of low-rank component coefficients
%   b: p-by-1 vector of high-dim component coefficients
%   k: dimensionality of low-rank latent space
%   initZ: initial value for Z (optional)
%   initW: initial value for W (optional)
%   option: struct of optimization parameters (optional)
%
% Outputs:
%   Z: n-by-k matrix of low-rank component features
%   W: n-by-p matrix of high-dim component features
%   objVals: objective value at each iteration

"""
import numpy as np


class Hybrid_SL_REFIT:
    def __init__(self, seed=np.random.randint(1e4), verbose=0, veryverbose=0,
                 outeriter=500, inneriter=5000, outertol=1e-4, innertol=1e-6):
        self.seed = seed
        self.verbose = verbose
        self.veryverbose = veryverbose
        self.outeriter = outeriter
        self.inneriter = inneriter
        self.outertol = outertol
        self.innertol = innertol

    def hybrid_sl_refit_(self, X, A, b, k, initZ=[], initW=[]):
        # 获得维度
        n, p = X.shape

        # 初始化变量
        Z, W = self.init_variables(n, k, p, initZ, initW)

        # 进行优化
        Z_, W_, objValsZW = self.optimize_ZW(X, Z, A, W, b)

        return Z_, W_, objValsZW

    def init_variables(self, n, k, p, initZ, initW):

        # 设置随机种子
        np.random.seed(self.seed)

        # 初始化 Z
        if initZ == []:
            Z = np.random.randn(n, k)  # 默认生成标准正态分布的随机数np.random.randn（n,p）
            Z = self.lF_projrct(Z)
        else:
            Z = initZ

        # 初始化 W
        if initW == []:
            W = np.random.randn(n, p)
            W = self.lF_projrct(W)
        else:
            W = initW

        return Z, W

    # 联合优化{Z, W}
    def optimize_ZW(self, X, Z, A, W, b):

        # 存储目标函数值
        objVals = np.zeros(self.inneriter)

        # 设置初始步长
        alpha = 1.0

        #  初始化起始值
        Zext = Z
        Wext = W
        theta = 1

        # 进行优化
        for iter1 in range(self.inneriter):
            # 计算梯度
            Zgrad = -np.dot((X - np.dot(Zext, A) - Wext * b.T), A.T)
            Wgrad = -(X - np.dot(Zext, A) - Wext * b.T) * b.T

            # 计算初始损失值
            gCurr = (1 / 2) * np.linalg.norm(X - np.dot(Zext, A) - Wext * b.T) ** 2
            # 计算新的 W， A
            for ls_it in range(100):
                Zplus = Zext - alpha * Zgrad
                Znew = self.lF_projrct(Zplus)
                Zgengrad = (Zext - Znew) / alpha
                Wplus = Wext - alpha * Wgrad
                Wnew = self.lF_projrct(Wplus)
                Wgengrad = (Wext - Wnew) / alpha
                gNew = (1 / 2) * np.linalg.norm(X - np.dot(Znew, A) - Wnew * b.T) ** 2
                if (gNew <= gCurr - alpha * np.sum((Zgrad * Zgengrad)) - alpha * np.sum(Wgrad * Wgengrad)
                        + .5 * alpha * np.sum((Zgengrad ** 2)) + .5 * alpha * np.sum(
                            Wgengrad ** 2)):  # A的所有元素之和 A.sum()
                    break
                else:
                    alpha = .5 * alpha
            # 计算新的Zext, Wext, theta
            thetanew = (1 + np.sqrt(1 + 4 * theta ** 2)) / 2
            Zext = Znew + (Znew - Z) * (theta - 1) / (thetanew)
            Wext = Wnew + (Wnew - W) * (theta - 1) / (thetanew)
            # 存储新的Z, b, theta
            Z = Znew
            W = Wnew
            theta = thetanew
            # 存储新的目标值和损失值
            objVals[iter1] = gNew
            # 收敛性判断
            if (iter1 >= 10) & ((abs(objVals[iter1] - objVals[iter1 - 1]) /
                                 max(1, np.abs(objVals[iter1 - 1]))) < self.innertol):
                break
            # 打印目标函数值
            if self.veryverbose:
                print('Optimizing Zb: Inner Iter {}: Obj = {}\n'.format(iter1, objVals[iter1]))

        return Z, W, objVals

    def lF_projrct(self, w):
        w_ = w / max(1, np.linalg.norm(w))
        return w_

    def l1_prox(self, bplus, u):
        bnew = np.sign(bplus) * np.maximum(np.abs(bplus) - u, 0)
        return bnew

    def l2_pro_mex(self, Aplus, u, n):
        a = np.linalg.norm(Aplus, ord=n)
        Anew = Aplus * np.maximum(0, a - u) / a
        return Anew


if __name__ == "__main__":
    C = Hybrid_SL_REFIT()
    X = np.random.randn(100, 200)
    # Z = np.random.randn(100, 20)
    A = np.random.rand(20, 200)
    #     W = np.random.randn(100, 200)
    b = np.random.rand(200, )
    k = 20
    # Z, W = C.init_variables(100,20,200,initZ = [], initW = [])
    # Z1, W1, objVals = C.optimize_ZW(X, Z, A, W, b)
    Z, W, objValsZW = C.hybrid_sl_refit_(X, A, b, k, initZ=[], initW=[])
    print(objValsZW)
