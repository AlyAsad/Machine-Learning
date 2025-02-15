import pickle
import matplotlib.pyplot as plt
from pca import PCA
from autoencoder import AutoEncoder
from sklearn.manifold import TSNE
from umap import UMAP

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))






## MY CODE STARTS HERE





#function that plots the models for each dataset
def plotModel(x, y, method, datasetNumber):
    
    plt.scatter(x, y, alpha = 0.75)
    plt.title(f"{method} method with dataset {datasetNumber}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha = 0.25)
    plt.show()




##running for each type of dimensionality reduction method


####################################

# HYPERPARAMETERS: from any dimensions to 2 dimensions (only one parameter)

#first PCA (dataset1)
model = PCA(2)
model.fit(dataset1)
data = model.transform(dataset1)
plotModel(data[:, 0], data[:, 1], "PCA", 1)

#PCA for dataset 2
model = PCA(2)
model.fit(dataset2)
data = model.transform(dataset2)
plotModel(data[:, 0], data[:, 1], "PCA", 2)




####################################

# HYPERPARAMETERS: from dataset dims to 2 dims, lr = 0.001, epochs = 10000

#second AutoEncoder (dataset1)
model = AutoEncoder(dataset1.shape[1], 2, 0.001, 10000)
model.fit(dataset1)
data = model.transform(dataset1)
plotModel(data[:, 0], data[:, 1], "AutoEncoder", 1)

#AutoEncoder for dataset2
model = AutoEncoder(dataset2.shape[1], 2, 0.001, 10000) 
model.fit(dataset2)
data = model.transform(dataset2)
plotModel(data[:, 0], data[:, 1], "AutoEncoder", 2)



####################################

# HYPERPARAMETERS: default, from any dims to 2 dims

#third TSNE (dataset1)
model = TSNE(n_components = 2)
data = model.fit_transform(dataset1)
plotModel(data[:, 0], data[:, 1], "TSNE", 1)

#TSNE for dataset2
model = TSNE(n_components = 2)
data = model.fit_transform(dataset2)
plotModel(data[:, 0], data[:, 1], "TSNE", 2)

####################################

# HYPERPARAMETERS: number of neighbours = 10, from any dims to 2 dims

#fourth UMAP (dataset1)
model = UMAP(n_neighbors=10, n_components=2)
data = model.fit_transform(dataset1)
plotModel(data[:, 0], data[:, 1], "UMAP", 1)

#UMAP for dataset2
model = UMAP(n_neighbors=10, n_components=2)
data = model.fit_transform(dataset2)
plotModel(data[:, 0], data[:, 1], "UMAP", 2)