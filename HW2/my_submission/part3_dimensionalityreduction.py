import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt

# The dataset is already preprocessed...
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))


## MY CODE STARTS HERE


#function that plots the models for each dataset
def plotModel(x, y, method):
    
    plt.scatter(x, y, alpha = 0.75)
    plt.title(f"{method} method")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha = 0.25)
    plt.show()
    


# PCA first
method = PCA(n_components=2)
data = method.fit_transform(dataset)
plotModel(data[:, 0], data[:, 1], "PCA")



# TSNE second
method = TSNE(n_components=2)
data = method.fit_transform(dataset)
plotModel(data[:, 0], data[:, 1], "TSNE")



# UMAP third
method = UMAP(n_neighbors=100, n_components=2)
data = method.fit_transform(dataset)
plotModel(data[:, 0], data[:, 1], "UMAP")