import pickle
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# The dataset is already preprocessed...
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))



### MY CODE STARTS HERE

##################################################
# first we do everything related to HAC

print("Running HAC:")


# plotting dendrogram (code copied from the sample recitation code)
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)





# HYPERPARAMETERS:
linkages = ["single", "complete"]
distanceMetrics = ["euclidean", "cosine"]

Kvals = [2, 3, 4, 5]


bestSil = 0
bestLink = None
bestDistMetric = None
bestK = None

# grid search of hyperparameters:
for link in linkages:
    for distMetric in distanceMetrics:
        
        silVals = []
        
        print(f"HAC with hyperparameters: [{link}, {distMetric}]")
        
        # running the algo and plotting the dendrogram
        hac = AgglomerativeClustering(n_clusters=None, metric=distMetric, linkage=link, distance_threshold=0)
        hac.fit(dataset)
        plot_dendrogram(hac, truncate_mode="lastp", p = 100)
        plt.title(f"Dendrogram for {link}, {distMetric}")
        plt.show()
        
        
        # now for the silhouette analysis
        for K in Kvals:
            hac = AgglomerativeClustering(n_clusters = K, metric=distMetric, linkage=link)
            labels = hac.fit_predict(dataset)
            silhouette_avg = silhouette_score(dataset, labels)
            silVals.append(silhouette_avg)
        
        #finding and printing best for this configuration
        bestSilVal = max(silVals)
        bestSilValIndex = silVals.index(bestSilVal) + 2
        print(f"Best K: {bestSilValIndex} with value: {bestSilVal:.3f} \n")
        
        #finding best over all configs
        if (bestSilVal > bestSil):
            bestSil = bestSilVal
            bestLink = link
            bestDistMetric = distMetric
            bestK = bestSilValIndex
        
        plt.plot(range(2, 6), silVals, marker = 'o')
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Silhouette score")
        plt.title(f"Silhouette scores for {link}, {distMetric}")
        plt.show()
        
        

#printing the best:
print("#####")
print("Best configuration:")
print(f"[{bestLink}, {bestDistMetric}, {bestK}], Silhouette value: {bestSil:.4f}")


print("\n\n ###################\n\n Running DBSCAN:")










##################################################
# now, we do everything related to DBSCAN



resultSils = []
resultConfigs = []
resultData = []
resultLabels = []

#defining hyperparameters
epsVals = [0.05, 0.1, 0.2] #removed 0.5, 1
distMetrics = ['euclidean', 'cosine'] #removed manhattan
minSampleVals = [1, 2, 4] #removed 3, 5

configNumber = 1
## grid searching, running 18 different configurations
for eps in epsVals:
    for distMetric in distMetrics:
        for minSamples in minSampleVals:
            
            print(f"{configNumber}) Testing configuration: [{eps, distMetric, minSamples}]")
            configNumber += 1
            
            dbscan = DBSCAN(eps=eps, min_samples=minSamples, metric=distMetric)
            predicted = dbscan.fit_predict(dataset)
            
            # removing outliers
            cluster_point_indices = []
            for i in range(len(predicted)):
                if predicted[i] != -1:
                    cluster_point_indices.append(i)
                    
            number_of_clusters = len(set(predicted)) - (1 if -1 in predicted else 0)
            
            clustered_points = dataset[cluster_point_indices]
            cluster_labels = predicted[cluster_point_indices]
            
            if number_of_clusters < 2:
                print(f"Not enough clusters, total clusters: {number_of_clusters}\n")
                continue
            elif number_of_clusters == len(clustered_points):
                print("Each point is a cluster.\n")
                continue
            
            #calculating silhouette if more than 1 cluster
            SilVal = silhouette_score(clustered_points, cluster_labels)
            resultSils.append(SilVal)
            resultConfigs.append([eps, distMetric, minSamples, number_of_clusters])
            resultData.append(clustered_points)
            resultLabels.append(cluster_labels)
            
            print(f"Clusters: {number_of_clusters}, Silhouette score: {SilVal:0.3f}\n")



##########


## now selecting best four configs

print("\n##############\nBest four configurations:\n")

resultIndexes = np.argsort(resultSils)
resultSils = [resultSils[i] for i in resultIndexes][-4:]
resultConfigs = [resultConfigs[i] for i in resultIndexes][-4:]
resultData = [resultData[i] for i in resultIndexes][-4:]
resultLabels = [resultLabels[i] for i in resultIndexes][-4:]


for i in range(4):
    
    sil = resultSils[i]
    config = resultConfigs[i]
    K = int(config[3])
    
    print(f"{4 - i}) eps = {config[0]}, dist metric = {config[1]}, min_samples = {config[2]}")
    print(f"Clusters formed: {K}, Silhouette score: {sil:.4f}\n")
    
    #plotting the data vs silhouette
    sil_scores = silhouette_samples(resultData[i], resultLabels[i], metric= config[1])
    
    
    plt.scatter(range(len(sil_scores)), sil_scores, c = resultLabels[i])
    plt.axhline(y = sil, color = "red")
    plt.title("Scatter Plot of Dataset Index vs Silhouette Score")
    plt.xlabel("Dataset Index")
    plt.ylabel("Silhouette Score")
    plt.show()