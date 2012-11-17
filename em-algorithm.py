import scipy as sp
import numpy as np
import random as rd
import scipy.cluster.vq as vq
from scipy.linalg import det, inv
import matplotlib.pyplot as plt

def gauss_mixture_calculate(x, u, sigma):
    D = len(x)
    x, u = sp.asarray(x), sp.asarray(u)
    y = x-u
    return sp.exp(-(sp.dot(y, sp.dot(inv(sigma), y)))/2.0) / (((2*sp.pi)**(D/2.0)) * (det(sigma) ** 0.5))


def kmeans(X, K):
    """
    kmeans to find clusers:
    x: dataset
    k: num of clusters
    #todo: this implements just for 2 cluster, initilization need to re-implement
    """
    ret = {'mean':[], 'cov':[], 'coff':[]}
    kmean_ret = vq.kmeans(X, K)  
    
    ##assign data to cluster to calculate covariance
    data = []
    
    for i in range(0,K,1):
      data.append([])

    for i in range(0,X.shape[0],1):
      dis = []
      max = 0
      max_idx = -1
      for j in range(0,K,1):
        _dis = ((X[i] - kmean_ret[0][j])**2).sum()
        if _dis >= max:
          max = _dis
          max_idx = j  
      data[max_idx].append(X[i]) 
   
    for i in range(0,K,1):
      data[i] = np.asarray(data[i]) 
      ret['cov'].append(np.cov(data[i].transpose()))
      ret['mean'].append(kmean_ret[0][i])
      ret['coff'].append(float(data[i].size/2)/X.shape[0]) 
    
    return ret


def main_loop(init_param, X, K, iter=1000, tol=1e-6):
    """
    Gaussian Mixture Model
    Arguments:
    - `X`: Input data (2D array, [[x11, x12, ..., x1D], ..., [xN1, ... xND]]).
    - `K`: Number of clusters.
    - `iter`: Number of iterations to run.
    - `tol`: Tolerance.
    """
    X = sp.asarray(X)
    N, D = X.shape
    pi = sp.asarray(init_param['coff'])
    mu = sp.asarray(init_param['mean'])
    sigma = sp.asarray(init_param['cov'])
    
    L = sp.inf

    for i in xrange(iter):
        # E-step
        gamma = sp.apply_along_axis(lambda x: sp.fromiter((pi[k] * gauss_mixture_calculate(x, mu[k], sigma[k]) for k in xrange(K)), dtype=float), 1, X)
        gamma /= sp.sum(gamma, 1)[:, sp.newaxis]

        # M-step
        Nk = sp.sum(gamma, 0)
        mu = sp.sum(X*gamma.T[..., sp.newaxis], 1) / Nk[..., sp.newaxis]
        xmu = X[:, sp.newaxis, :] - mu
        sigma = sp.sum(gamma[..., sp.newaxis, sp.newaxis] * xmu[:, :, sp.newaxis, :] * xmu[:, :, :, sp.newaxis], 0) / Nk[..., sp.newaxis, sp.newaxis]
        pi = Nk / N

        # Likelihood
        Lnew = sp.sum(sp.log2(sp.sum(sp.apply_along_axis(lambda x: sp.fromiter((pi[k] * gauss_mixture_calculate(x, mu[k], sigma[k]) for k in xrange(K)), dtype=float), 1, X), 1)))
        if abs(L-Lnew) < tol: break
        L = Lnew
        print "log likelihood=%s" % L

    return dict(pi=pi, mu=mu, sigma=sigma, gamma=gamma)


if __name__ == '__main__':
    data = sp.loadtxt("data10000.txt", delimiter=',', unpack=True)
    data = sp.transpose(data)
    K = 2
    clusters = kmeans(data, K) 
    d = main_loop(clusters, data, K)

    print "pi=%s\nmyu=%s\nsigma=%s" % (d['pi'], d['mu'], d['sigma'])
    gamma = d['gamma']

    plt.scatter(data[:, 0][gamma[:, 0] >= 0.5], data[:, 1][gamma[:, 0] >= 0.5], color='r')
    plt.scatter(data[:, 0][gamma[:, 1] > 0.5 ], data[:, 1][gamma[:, 1] > 0.5 ], color='g')
    plt.show()
