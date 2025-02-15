import pickle
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import f1_score


dataset, labels = pickle.load(open("../datasets/part3_dataset.data", "rb"))



### MY CODE STARTS HERE



# first, I define the outer and inner cross-validation folds
outerFolds = RepeatedStratifiedKFold(n_splits = 3, n_repeats = 5, random_state = 2547875) # random state is my student ID hehe
innerFolds = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 2547875) # random state is my student ID hehe




# Now, I define the classifiers to test and their hyperparameters.
# list of tuples, first element of a tuple is the name of classifier, 
# second element is classifier function,
# third element is the possible hyperparameters as a dictionary

classifiers = [
    ("KNN", KNeighborsClassifier(), {"kneighborsclassifier__n_neighbors": [3, 5, 7], "kneighborsclassifier__weights": ["uniform", "distance"]}), # KNN, 6 configs
    ("SVM", SVC(), {"svc__C": [0.1, 1, 10], "svc__kernel": ["rbf", "sigmoid"]}), # SVM, 6 configs
    ("Decision Tree", DecisionTreeClassifier(), {"decisiontreeclassifier__criterion": ["gini", "entropy"], "decisiontreeclassifier__max_depth": [5, 50]}), # decision tree, 4 configs
    ("Random Forest", RandomForestClassifier(), {"randomforestclassifier__n_estimators": [100, 200], "randomforestclassifier__criterion": ["gini", "entropy"]}), # random forest, 4 configs
    ("MLP", MLPClassifier(), {"mlpclassifier__hidden_layer_sizes": [(20,), (30,)], "mlpclassifier__activation": ["tanh", "relu"], "mlpclassifier__learning_rate_init": [0.1, 1]}), # MLP, 8 configs
    ("Gradient Boosting", GradientBoostingClassifier(), {"gradientboostingclassifier__loss": ["log_loss"], "gradientboostingclassifier__learning_rate": [0.01, 0.1]}) # gradient boosting, 2 configs
]


## helper function to run stochastic models x amount of times for best results
def stochasticGridSearch(gridSearch, trainData, trainLabels, attempts = 10):
    
    accuracies = []
    deviations = []
    firstRun = True
    # basically running it attempts (10) times, and saving the best accuracy
    for i in range(attempts):
        params, newAccuracies, stds = nonStochasticGridSearch(gridSearch, trainData, trainLabels)
        
        if firstRun:
            firstRun = False
            accuracies = newAccuracies
            deviations = stds
        else:
            for i in range(len(accuracies)):
                if newAccuracies[i] > accuracies[i]:
                    accuracies[i] = newAccuracies[i]
                    deviations[i] = stds[i]
    
    return params, accuracies, deviations


## helper function to run nonstochastic models once
def nonStochasticGridSearch(gridSearch, trainData, trainLabels):
    
    gridSearch.fit(trainData, trainLabels)
    
    results = gridSearch.cv_results_
    
    params = []
    accuracies = []
    stds = []
    
    # now, for each hyperparameter configuration tested, we save its accuracy and std
    for param, mean, std in zip(results["params"], results["mean_test_score"], results["std_test_score"]):
        
        params.append(param)
        accuracies.append(mean)
        stds.append(std)
    
    return params, accuracies, stds


# MAIN LOOP TO TRAIN AND EVALUATE EACH CLASSIFIER
for classifierName, classifierFunc, classifierParams in classifiers:
    
    print("\n########################################################### \n")
    print(f"Evaluating {classifierName}:")
    startTime = time.time()
    
    # setting up the pipeline, to normalize our data
    pipeModel = make_pipeline(
        MinMaxScaler(feature_range = (-1, 1)),
        classifierFunc
    )
    
    # setting up the gridsearch model
    gridSearch = GridSearchCV(
        estimator = pipeModel,
        param_grid = classifierParams,
        scoring = "f1_micro",
        cv = innerFolds,
        n_jobs = -1     # for speed, it will use all processors
    )
    
    
    
    f1Scores = []
    
    
    firstRun = True
    ## now, for the outer cross-validation loop
    for train_indices, test_indices in outerFolds.split(dataset, labels):
        
        # splitting the data and labels
        trainData, testData = dataset[train_indices], dataset[test_indices]
        trainLabels, testLabels = labels[train_indices], labels[test_indices]
        
        
        # if model is stochastic, we need to run it multiple times and get the best results for each hyperparameter config
        if classifierName in ["Random Forest", "MLP", "Gradient Boosting"]:
            params, accuracies, stds = stochasticGridSearch(gridSearch, trainData, trainLabels, attempts = 10)
            
        # else model is not stochastic
        else:
            params, accuracies, stds = nonStochasticGridSearch(gridSearch, trainData, trainLabels)
        
        
        # if this is the first run, print hyperparameter search results
        if firstRun:
            firstRun = False
            i = 0
            for param, mean, std in zip(params, accuracies, stds):
                i += 1
                print(f"{i}) config: {param}")
                print(f"Average F1 score: {mean:.4f}, std: {std:.4f}")
        
        
        
        # now, i select the best hyperparameter config, and note its f1 score on testing dataset of outer fold
        bestScore = 0
        bestParam = {}
        for param, mean in zip(params, accuracies):
            if mean > bestScore:
                bestScore = mean
                bestParam = param
        
        
        # making the final model to test on outer fold
        outerModel = make_pipeline(
            MinMaxScaler(feature_range = (-1, 1)),
            classifierFunc
        )
        
        outerModel.set_params(**bestParam)
        outerModel.fit(trainData, trainLabels)
        predicted = outerModel.predict(testData)
        
        outerScore = f1_score(testLabels, predicted, average = "micro")
        f1Scores.append(outerScore)
    
    
    
    
    # calculating mean and conf for this classifier
    f1ScoresMean = np.mean(f1Scores)
    f1ScoresConf = 1.96 * np.std(f1Scores) / np.sqrt(len(f1Scores))
    
    # printing final results
    print(f"\nF1 score for this configuration: {f1ScoresMean:.4f} {u"\u00B1"} {f1ScoresConf:.4f}")
    print(f"Time taken: {(time.time() - startTime):.2f} seconds")
