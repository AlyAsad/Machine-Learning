import math

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters
    
    ## MY CODE STARTS HERE
        
    def predict(self, instance):
        
        data_shape = self.dataset.shape
        min_dist = [math.inf] * self.K
        min_labels = [None] * self.K
        
        maxDist = math.inf
        maxDistIndex = 0
        
        ## finding the nearest neighbours
        for i in range(data_shape[0]):
            currDist = self.similarity_function(instance, self.dataset[i], self.similarity_function_parameters)
            if currDist < maxDist:
                min_dist[maxDistIndex] = currDist
                min_labels[maxDistIndex] = self.dataset_label[i]
                maxDist, maxDistIndex = maxDistAndIndex(min_dist)
        
        
        ## classifying the instance based on NNs, and returning it
        return max(min_labels, key=min_labels.count)
    
    ## MY CODE ENDS HERE



## MY CODE

def maxDistAndIndex(arr):
        maxDist = arr[0]
        ind = 0
        for i in range(1, len(arr)):
            if arr[i] > maxDist:
                maxDist = arr[i]
                ind = i
        
        return maxDist, ind
