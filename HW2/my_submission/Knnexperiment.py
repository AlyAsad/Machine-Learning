import pickle
from Distance import Distance
from Knn import KNN
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))



## MY CODE STARTS HERE


## global variables
bestMean = 0
bestDev = 0
bestK = 0
bestFun = 0


# defining the sklearn kfold instance, 10 folds, repeated 5 times
rskf = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 5, random_state = 0)


## defining hyperparameters (total of 12 configurations)
Kvalues = [1, 5, 7, 9]
similarityFunction = ["Cosine", "Minkowski", "Mahalanobis"]


param = 0
## looping over hyperparameters for crossvalidation (12 times)
for K in Kvalues:
    for fun in similarityFunction:
        param += 1
        
        print(str(param) + ") Testing hyperparameters: K =",str(K) + ", Distance function:", fun)
        
        accuracies = [0, 0, 0, 0, 0]
        
        i = 0
        # for each fold/repetition
        for train_index, test_index in rskf.split(dataset, labels):
            run = int(i / 10)
                
            #separating data
            train_data, train_labels = dataset[train_index], labels[train_index]
            test_data, test_labels =  dataset[test_index], labels[test_index]
            
            #setting the input parameters
            if fun == "Cosine":
                funInput = Distance.calculateCosineDistance
                funParam = None
            elif fun == "Minkowski":
                funInput = Distance.calculateMinkowskiDistance
                funParam = 2
            else:
                funInput = Distance.calculateMahalanobisDistance
                funParam = np.linalg.inv(np.cov(m = train_data, rowvar = False))
            
            #initializing my knn model
            knn = KNN(train_data, train_labels, funInput, funParam, K)
            
            #running the model on the test data
            predictions = [knn.predict(test) for test in test_data]
            
            #finding accuracy for this run
            accuracies[run] += 10*np.mean(predictions == test_labels)
            i += 1
        
        #printing confidence interval    
        mean = np.mean(accuracies)
        deviation = 1.96 * (np.std(accuracies)/np.sqrt(len(accuracies)))
        print("Confidence interval: %.2f" %mean, "\u00B1 %.3f" %deviation, "\n")
        
        if mean > bestMean:
            bestMean = mean
            bestDev = deviation
            bestK = K
            bestFun = fun
        
            


print("\nBest configuration: K =",str(bestK) + ", Distance function:", bestFun)
print("Confidence interval: %.2f" %bestMean, "\u00B1 %.3f" %bestDev)