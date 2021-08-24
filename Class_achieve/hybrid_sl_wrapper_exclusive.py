"""
% Function to select the best lambda for hybrid SL by maximizing the fit
% of the true low-rank subspace
%
% Inputs:
%   X: observed data matrix
%   k: true low-rank dimension
%   L_true: true low-rank component
%   S_true: true high-rank component
% Outputs:
%   bestZ: best estimate of low-rank features
%   bestA: best estimate of low-rank coefficients
%   bestW: best estimate of high-dim features
%   bestb: best estimate of high-dim coefficients
%   bestErr: best low-rank subspace error

"""

import numpy as np
from Hybrid_SL_ACCE import Hybrid_SL_ACCE

def hybrid_sl_wrapper_exclusive(X, k, lambda_, verbose=0):
    gamma = 0
    Z = []
    A = []
    W = []
    b = []

    # initialize storage for objective values
    objVals = []
    gammaVals = []

    #  initialize storage for AIC scores
    aicVals = [];

    # learn a sequence of models with increasing gamma
    while (True):

        # run optimization with current value of gamma, initialize with previous solution

        Z, A, W, b, currObjVals, objIds = Hybrid_SL_ACCE().hybrid_sl_optimization_accel(X, k, gamma, 0, lambda_, Z, A,
                                                                                        W, b);
        objVals.append(currObjVals)
        gammaVals.append(gamma * np.ones(len(currObjVals), ))

        # calculate density of A and b
        indOnA = sum(np.abs(A)) > 0
        indOnB = np.abs(b.T) > 0
        dnsA = 100 * np.mean(indOnA)
        dnsB = 100 * np.mean(indOnB)
        dnsBoth = 100 * np.mean(indOnA & indOnB)
        dnsEither = 100 * np.mean(indOnA | indOnB)

        # calculate AIC score
        p1 = sum(sum(np.abs(A)) != 0)
        p2 = sum(np.abs(b) != 0)
        numParam = k * (k - 1) / 2 + k * p1 + 1 + p2 * (p2 - 1) / 2 + p2
        lossVal = sum(sum((X - np.dot(Z, A) - np.dot(W, np.diag(b))) ** 2)) / 2
        aicVals.append(2 * numParam + 2 * lossVal)

        # print message
        if verbose:
            print('Gamma = %g: Dens of A = %.1f%%, Dens of b = %.1f%%, Overlap = %.1f%%, Coverage = %.1f%%\n' \
                  % (gamma, dnsA, dnsB, dnsBoth, dnsEither))

        #  check stopping criterion
        if np.sum(dnsBoth) == 0:
            break

        # update gamma
        if gamma == 0:
            gamma = .25
        else:
            gamma = 2 * gamma

    return Z, A, W, b, objVals, gammaVals, aicVals


if __name__ == '__main__':
    np.random.seed(1222)
    np.random.seed(222)
    X = np.random.rand(100, 200)
    k = 20
    lambda_ = .1
    Z, A, W, b, objVals, gammaVals, aicVals = hybrid_sl_wrapper_exclusive(X, k, lambda_)
    print(aicVals)
