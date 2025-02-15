import numpy as np

class PCA:
    def __init__(self, projection_dim: int):
        """
        Initializes the PCA method
        :param projection_dim: the projection space dimensionality
        """
        self.projection_dim = projection_dim
        # keeps the projection matrix information
        self.projection_matrix = None
        
    def fit(self, x: np.ndarray) -> None:
        """
        Applies the PCA method and obtains the projection matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should assign the resulting projection matrix to self.projection_matrix
        """
        
        #i am centering the data around 0
        x = x - np.mean(x, axis = 0)
        
        #calc covariance matrix
        x = np.cov(x, rowvar = False)
        
        #finding eigenvals and eignevectors now
        _, eigenVecs = np.linalg.eigh(x)
        
        #since eigh() already sorts the vals, no need to sort, just take last N values
        self.projection_matrix = eigenVecs[:, -self.projection_dim:]
        
        
        

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        After learning the projection matrix on a given dataset,
        this function uses the learned projection matrix to project new data instances
        :param x: data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        this function should utilize self.projection_matrix for the operations
        """
        
        # just multiplying projection matrix with input to reduce its dimension
        x = x - np.mean(x, axis = 0)
        return np.matmul(x, self.projection_matrix)
