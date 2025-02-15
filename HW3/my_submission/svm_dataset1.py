import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset, labels = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))




# copied the plotting code from the recitation files

def model_display_boundary(X, model, label):
    h = .01  # step size in the mesh, we can decrease this value for smooth plots, i.e 0.01 (but ploting may slow down)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 3, X[:, 0].max() + 3
    y_min, y_max = X[:, 1].min() - 3, X[:, 1].max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    aa = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(aa)
    Z = Z.reshape(xx.shape)
    # plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.contourf(xx, yy, Z, alpha=0.25) # cmap="Paired_r",
    # plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=label, cmap="Paired_r", edgecolors='k');
    x_ = np.array([x_min, x_max])

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()
    
    
    
    
    
### MY CODE STARTS HERE




# defining the hyperparameters to test, their cross product will be tested
# TOTAL OF 9 CONFIGURATIONS WILL BE TESTED
Kernels = ['poly', 'rbf', 'sigmoid']    # default degree for poly (3)
Cvals = [0.1, 1.0, 10.0]





i = 1

# grid search over hyperparameters
for kernel in Kernels:
    for cVal in Cvals:
        
        # printing for neat output
        print(f"{i}) Running hyperparameters [Kernel: {kernel}, C: {cVal}]")
        i += 1
        
        
        # defining the model and training it
        svm = SVC(C = cVal, kernel = kernel)
        svm.fit(dataset, labels)
        
        # getting the accuracy and printing it
        acc = accuracy_score(labels, svm.predict(dataset)) * 100
        print(f"Accuracy: {acc:.2f}%\n")

        
        # plotting the model
        plt.title(f"SVM with Kernel: {kernel}, C: {cVal}")
        plt.xlabel("x")
        plt.ylabel("y")
        model_display_boundary(dataset, svm, labels)