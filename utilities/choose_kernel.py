import numpy as np
from scipy.spatial.distance import cdist

def choose_kernel(kernelType = "gaussianKernel"):
    d = kernelType
    if kernelType == "gaussianKernel":
        return GaussianKernel()
    else:
        print("No such kernel is implemented yet")

class GaussianKernel():
    def __init__(self):
        return

    def kernel_operator(self, X2, bandwidth, factor=None):
        return lambda x:  (1 / len(X2)) * self.calcKernel(x, X2, bandwidth, factor=factor)

    def calcKernel(self, X1, X2, bandwidth, factor=None):
        self.bandwidth = bandwidth
        if factor != None:
            bandwidth = factor * bandwidth

        D = cdist(X1, X2, metric = 'sqeuclidean')
        D = (-1 / (2*bandwidth ** 2)) * D
        return np.exp(D)

    def prediction(self, Xts, Xtr, coef):
        Knn = self.calcKernel(Xts, Xtr, self.bandwidth)
        return np.dot(Knn, coef)

    def calc_derivative(self, alpha, X1, X2):

        d = (X1-X2)
        dd = (X2-X1)
        m = np.multiply(d, alpha)
        mm = np.multiply(dd, alpha)
        return -1/(self.bandwidth)**2 * np.dot(self.calcKernel(X1, X2, self.bandwidth), np.multiply((X1-X2), alpha))
