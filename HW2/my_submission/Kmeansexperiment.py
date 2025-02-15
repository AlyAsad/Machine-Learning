import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))



## MY CODE STARTS HERE



def solveAndPlot(dataset, setNo):
    lossAvgs, lossConfs = [], []
    silAvgs, silConfs = [], []
    
    ## SOLVING
    for K in range(2, 11):
        lossVals = []
        silVals = []
        
        # running 10 times for avg loss/silhouette
        for _ in range(10):
            kmeans = KMeans(n_clusters = K, init="k-means++", n_init=10) # running 10 times and choosing the best inertia
            kmeans.fit(dataset)
            
            # updating loss
            lossVals.append(kmeans.inertia_)
            # updating silhouette
            silVals.append(silhouette_score(dataset, kmeans.labels_))
        
        # updating for graph
        lossM = np.mean(lossVals)
        lossAvgs.append(lossM)
        lossDev = 1.96 * np.std(lossVals)/np.sqrt(len(lossVals))
        lossConfs.append(lossDev)
        
        silM = np.mean(silVals)
        silAvgs.append(silM)
        silDev = 1.96 * np.std(silVals)/np.sqrt(len(silVals))
        silConfs.append(silDev)
        
        print("K = %d: avg loss: %.2f \u00B1 %.4f, avg silhouette: %.3f, \u00B1 %.4f" %(K, lossM, lossDev, silM, silDev))
    
    #newline for cleanliness
    print()
    
    
    ## PLOTTING
    plt.plot(range(2,11), lossAvgs)
    plt.errorbar(range(2,11), lossAvgs, yerr = lossConfs, fmt = 'o', capsize=5)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Loss")
    plt.title("K vs Loss graph for dataset %d" %setNo)
    plt.show()
    
    plt.plot(range(2,11), silAvgs)
    plt.errorbar(range(2,11), silAvgs, yerr = silConfs, fmt = 'o', capsize=5)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("K vs Silhouette graph for dataset %d" %setNo)
    plt.show()

    



print("Solving dataset 1 with Kmeans++:")
solveAndPlot(dataset1, 1)

print("Solving dataset 2 with Kmeans++:")
solveAndPlot(dataset2, 2)