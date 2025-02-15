import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataset, labels = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

### MY CODE STARTS HERE



# defining the hyperparameters to test, their cross product will be tested
# TOTAL OF 9 CONFIGURATIONS WILL BE TESTED
# default degree for poly is 3
param_grid = {'svm__kernel': ['poly', 'rbf', 'sigmoid'], 'svm__C': [0.1, 1, 10]}



# where the accuracies of each hyperparameter config will be stored as a list
params = [{}] * 9
accuracies = [[] for _ in range(9)]     # had to do this to avoid shallow copy


# performing repeated cross validation, 5 times
for i in range(5):
    
    print(f"Running cross-validation #{i + 1}")
    
    # creating the shuffle for the dataset
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
    
    
    # preprocessing the data
    pipeModel = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
    
    
    # grid-search on this shuffle
    gridSearch = GridSearchCV(
        estimator = pipeModel,
        param_grid = param_grid,
        scoring = "accuracy",
        cv = kfold,
        n_jobs = -1     # for speed, it will use all processors
    )
    
    gridSearch.fit(dataset, labels)
    results = gridSearch.cv_results_
    
    
    # now, updating the accuracies
    for i, (param, mean) in enumerate(zip(results["params"], results["mean_test_score"])):
        
        params[i] = param
        accuracies[i].append(mean * 100)
    



### now we have 5 accuracies for each of the 9 hyperparameters configs (9x5 array)
    # (due to 5 times cross-validiation)


print()
bestMean = 0.0
bestConf = 0.0
bestIndex = -1
bestKernel = ""
bestC = ""


# printing the mean accuracy and confidence interval for each configuration
for i, (param, means) in enumerate(zip(params, accuracies)):
    
    print(f"{i + 1}) Config: [C: {param["svm__C"]}, Kernel: {param["svm__kernel"]}]")
    
    # finding mean and confidence interval
    mean = np.mean(means)
    conf = 1.96 * np.std(means) / np.sqrt(len(means))
    
    # printing mean and confidence interval
    print(f"Average accuracy: {mean:0.2f} {u"\u00B1"} {conf:0.3f}\n")
    
    
    # updating best mean
    if mean > bestMean:
        bestMean = mean
        bestConf = conf
        bestIndex = i
        bestKernel = param["svm__kernel"]
        bestC = param["svm__C"]
    


## printing best configuration
print("###################################\n")
print(f"Best config: {bestIndex + 1}) [C: {bestC}, Kernel: {bestKernel}]")
print(f"Best accuracy: {bestMean:0.2f} {u"\u00B1"} {bestConf:0.3f}\n")