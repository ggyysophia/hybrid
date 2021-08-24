# 使用类实现函数的功能
"""
 Function to optimize the objective function for hybrid low-rank + sparse
 subspace learning using alternating accelerated proximal gradient descent.
Specifically, the problem is given by

   min_{Z,A,W,b} (1/2) * ||X - Z*A - W*diag(b)||_F^2
       + gamma * ||A*diag(b)||_{1,2}
       + lambda1 * ||A||_{1,2} + lambda2 * ||b||_1
   s.t. ||Z||_F <= 1, ||W||_F <= 1

 Inputs:
   X: n-by-p matrix of observed features
   k: dimensionality of low-rank latent space
   gamma: regularization parameter for joint penalty on A and b
   lambda1: regularization parameter for individual penalty on A
   lambda2: regularization parameter for individual penalty on b
   initZ: initial value for Z (optional)
   initA: initial value for A (optional)
   initW: initial value for W (optional)
   initb: initial value for b (optional)
   option: struct of optimization parameters (optional)

 Outputs:
   Z: n-by-k matrix of low-rank component features
   A: k-by-p matrix of low-rank component coefficients
   W: n-by-p matrix of high-dim component features
   b: p-by-1 vector of high-dim component coefficients
   objVals: objective value at each iteration
   objIds: indicator of which subroutine was running at each iteration

"""
import numpy as np


class Hybrid_SL:
    def __init__(self, seed=np.random.randint(1e4), verbose=0, veryverbose=0,
                 outeriter=500, inneriter=5000, outertol=1e-4, innertol=1e-6):
        self.seed = seed
        self.verbose = verbose
        self.veryverbose = veryverbose
        self.outeriter = outeriter
        self.inneriter = inneriter
        self.outertol = outertol
        self.innertol = innertol

    def hybrid_sl_optimization(self, X, k, gamma, lambda1, lambda2, initZ=[], initA=[], initW=[], initb=[]):
        # 获得维度
        n, p = X.shape

        # 初始化变量
        Z, A, W, b = self.init_variables(n, k, p, initZ, initA, initW, initb)

        # 初始化存储
        objVals = []
        objIds = []
        objValsOuter = np.zeros(self.outeriter)

        # 进行双目标函数优化
        for iter1 in range(self.outeriter):
            # 优化系数
            W, A, objValsWA = self.optimize_WA(X, Z, A, W, b, gamma, lambda1, lambda2)
            objVals.append(objValsWA)
            objIds.append(np.zeros((len(objValsWA), 1)))

            # 优化特征
            Z, b, objValsZb = self.optimize_Zb(X, Z, A, W, b, gamma, lambda1, lambda2)
            objVals.append(objValsZb)
            objIds.append(np.ones((len(objValsZb), 1)))

            # 计算目标函数
            # Wb   #Wb_ = np.array([W[:,i]*b[i] for i in range(p)]).T   等价于 W*b.T
            # Ab_ = np.array([A[:, i] * b[i] for i in range(p)]).T     等价于 A *b.T
            # 返回按列处理的l2范数  # https://blog.csdn.net/qq_35154529/article/details/82754157

            # A_ = np.linalg.norm(A, ord=2, axis=0, keepdims=False)
            # ord:1,2,00范数,默认l2范数； axis=0：按行求范数; Ab_ = np.sum(A_*b)  # 按元素相乘
            #  gamma*sum(sqrt(sum((bsxfun(@times,A,b.T)).^2,1))) = gamma * np.sqrt(sum(Ab_**2)).sum()
            # np.linalg.norm(b,ord=1) :l1范数，sum(np.abs(b))

            objValsOuter[iter1] = (1 / 2) * np.linalg.norm(X - np.dot(Z, A) - W * b.T) ** 2 + gamma * np.sum(np.sqrt(np.sum((A * b.T) ** 2, axis=0))) \
                                  + lambda1 * np.sum(np.sqrt(np.sum(A ** 2, axis=0))) + np.sum(np.abs(b))
            # 打印信息
            if self.verbose != 0:
                dnsA = 100 * sum(sum(np.abs(A)) > 0) / p  # python与matlab sum()相同,默认按列求和
                dnsB = 100 * sum(np.abs(b) > 0) / p
                print('Outer Iter:{} , Dens of A ={}, Dens of b = {}\n', iter1, objValsOuter[iter1], dnsA, dnsB)

            # 检查收敛性
            if (iter1 >= 5) & ((np.abs(objValsOuter[iter1] - objValsOuter[iter1 - 1]) /
                                max(1, np.abs(objValsOuter[iter1 - 1]))) <= self.outertol):
                break

        return Z, A, W, b, objVals, objIds

    def init_variables(self, n, k, p, initZ=[], initA=[], initW=[], initb=[]):
        # 设置随机种子
        np.random.seed(self.seed)

        # 初始化 Z
        if initZ == []:
            Z = np.random.randn(n, k)  # 默认生成标准正态分布的随机数np.random.randn（n,p）
            Z = self.lF_projrct(Z)
        else:
            Z = initZ

        # 初始化 A
        if initA == []:
            A = np.random.rand(k, p)  # 生成（0，1）之间的随机数
        else:
            A = initA

        # 初始化 W
        if initW == []:
            W = np.random.randn(n, p)
            W = self.lF_projrct(W)
        else:
            W = initW

        # 初始化 b
        if not initb:
            b = np.random.rand(p, )
        else:
            b = initb

        return Z, A, W, b

    # 联合优化{W, A}
    def optimize_WA(self, X, Z, A, W, b, gamma, lambda1, lambda2):
        # 存储目标函数值
        objVals = np.zeros(self.inneriter)

        # 计算和存储惩罚值
        pen3 = lambda1 * np.sum(np.abs(b))

        # 计算初始损失值
        gCurr = (1 / 2) * np.linalg.norm(X - np.dot(Z, A) - W * b.T) ** 2

        # 设置初始步长
        alpha = 1.0

        # 进行优化
        for iter1 in range(self.inneriter):
            # 计算梯度
            Wgrad = -(X - np.dot(Z, A) - W * b.T) * b.T
            Agrad = -np.dot(Z.T, (X - np.dot(Z, A) - W * b.T))
            # 计算新的 W， A
            for ls_it in range(100):
                Wplus = W - alpha * Wgrad
                Wnew = self.lF_projrct(Wplus)
                Wgengrad = (W - Wnew) / alpha
                Aplus = A - alpha * Agrad
                Anew = self.l2_pro_mex(Aplus, alpha * (gamma * np.abs(b) + lambda1).T, 2)  # 转置 np.transpose(data), data.T
                Agengrad = (A - Anew) / alpha
                gNew = (1 / 2) * np.linalg.norm(X - np.dot(Z, Anew) - Wnew * b.T) ** 2
                if (gNew <= gCurr - alpha * np.sum((Wgrad * Wgengrad)) - alpha * np.sum(Agrad * Agengrad) +
                        .5 * alpha * np.sum((Wgengrad ** 2)) + .5 * alpha * np.sum(Agengrad ** 2)):  # A的所有元素之和 A.sum()
                    W = Wnew
                    A = Anew
                    break
                else:
                    allpha = .5 * alpha
            # 存储新的目标值和损失值
            objVals[iter1] = gNew + gamma * np.sum(np.sqrt(np.sum((A * b.T) ** 2, axis=0))) + \
                             pen3 + lambda1 * np.sum(np.sqrt(np.sum(A ** 2, axis=0)))
            gCurr = gNew
            # 收敛性判断
            if (iter1 >= 10) & ((abs(objVals[iter1] - objVals[iter1 - 1]) /
                                 max(1, np.abs(objVals[iter1 - 1]))) < self.innertol):
                break
            # 打印目标函数值
            if self.veryverbose:
                print('Optimizing WA: Inner Iter {}: Obj = {}\n'.format(iter1, objVals[iter1]))

        return W, A, objVals

    # 联合优化{Z,b}
    def optimize_Zb(self, X, Z, A, W, b, gamma, lambda1, lambda2):
        # 存储目标函数值
        objVals = np.zeros(self.inneriter)

        # 计算和存储惩罚值
        pen2 = lambda1 * np.sum(np.sqrt(np.sum(A ** 2, axis=0)))

        # 计算初始损失值
        gCurr = (1 / 2) * np.linalg.norm(X - np.dot(Z, A) - W * b.T) ** 2

        # 设置初始步长
        alpha = 1.0

        # 进行优化
        for iter1 in range(self.inneriter):
            # 计算梯度
            Zgrad = -np.dot((X - np.dot(Z, A) - W * b.T), A.T)
            bgrad = -np.sum((X - np.dot(Z, A) - W * b.T) * W, axis=0).T
            # 计算新的 Z， b
            for ls_it in range(100):
                Zplus = Z - alpha * Zgrad
                Znew = self.lF_projrct(Zplus)
                Zgengrad = (Z - Znew) / alpha
                bplus = b - alpha * bgrad
                bnew = self.l1_prox(bplus, alpha * np.transpose(
                    (gamma * np.sqrt(np.sum(A ** 2, axis=0)) + lambda2)))  # 转置 np.transpose(data), data.T
                bgengrad = (b - bnew) / alpha
                gNew = (1 / 2) * np.linalg.norm(X - np.dot(Znew, A) - W * bnew.T) ** 2
                if (gNew <= gCurr - alpha * np.sum((Zgrad * Zgengrad)) - alpha * sum(bgrad * bgengrad)
                        + .5 * alpha * np.sum((Zgengrad ** 2)) + .5 * alpha * sum(bgengrad ** 2)):  # A的所有元素之和 A.sum()
                    Z = Znew
                    b = bnew
                    break
                else:
                    alpha = .5 * alpha
                # 存储新的目标值和损失值
            objVals[iter1] = gNew + gamma * np.sum(np.sqrt(np.sum((A * b.T) ** 2, axis=0))) + pen2 + lambda2 * np.sum(np.abs(b))
            gCurr = gNew
            # 收敛性判断
            if (iter1 >= 10) & ((abs(objVals[iter1] - objVals[iter1 - 1]) / max(1, np.abs(objVals[iter1 - 1]))) < self.innertol):
                break
            # 打印目标函数值
            if self.veryverbose:
                print('Optimizing Zb: Inner Iter {}: Obj = {}\n'.format(iter1, objVals[iter1]))

        return Z, b, objVals

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
    C = Hybrid_SL(seed=1111)
    X = np.random.randn(100, 200)
    Z, A, W, b = C.init_variables(100, 20, 200)
    # W1,A1,objVals = C.optimize_WA(X, Z, A, W, b, gamma = .1, lambda1 = .2,lambda2 = .3)
    # Z1,b1,objVals = C.optimize_Zb(X, Z, A, W, b, gamma = .1, lambda1 = .2,lambda2 = .3)
    Z1, A1, W1, b1, objVals, objIds = C.hybrid_sl_optimization(X, k=20, gamma=.1, lambda1=.2, lambda2=.3)
    print(Z1)


