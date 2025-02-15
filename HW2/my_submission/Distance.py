import numpy as np

class Distance:
    @staticmethod
    def calculateCosineDistance(x, y, _ = None):
        return 1 - (np.dot(x, y)/(np.sqrt(x.dot(x))*np.sqrt(y.dot(y))))
        
        
    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        return np.sum(np.abs(x - y) ** p) **(1/p)
        
        
    @staticmethod
    def calculateMahalanobisDistance(x,y, S_minus_1):
        xminusy = x - y
        dist2 = np.matmul(np.matmul(xminusy.T, S_minus_1), xminusy)
        return np.sqrt(dist2)
        
        
