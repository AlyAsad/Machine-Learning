import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, _, labels):
        """
        :param dataset: array of the data instances 
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """

        """
        Entropy calculations
        """
        
        ### dataset not used so made it _ for efficiency
        
        
        # finding out the proportions of each category in the labels
        _, prop = np.unique(labels, return_counts = True)
        props = prop / len(labels)
        
        # using the entropy formula from the slides
        ev = -np.sum(np.log2(props) * props)
        
        return ev

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        """
            Average entropy calculations
        """
        
        # im setting the convention that input attribute is integer index, not string
        
        # this is basically the values for this attribute from the dataset
        valsOfAttribute = dataset[:, attribute]
        
        # the length of these arrays is how many splits there will be, and each 
        # index states what the split's value is how many elements will go to that split
        splitVals, splitCounts = np.unique(valsOfAttribute, return_counts = True)
        
        
        avgE = 0.0
        # looping through each split
        for splitLabel, splitCount in zip(splitVals, splitCounts):
            
            # finding the indices of data which correspond to this split's value
            splitIndices = np.where(valsOfAttribute == splitLabel)[0]
            # finding the labels for data which correspond to this split
            splitLabels = labels[splitIndices]
            
            # finding entropy of this split
            entropy = self.calculate_entropy__([], splitLabels)
            
            # multiplying entropy with its proportion to add to avg entroy
            avgE += entropy * splitCount/len(valsOfAttribute)
        
        
        return avgE

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        """
            Information gain calculations
        """
        
        # just simple formula from slides
        return self.calculate_entropy__([], labels) - self.calculate_average_entropy__(dataset, labels, attribute)
        

    def calculate_intrinsic_information__(self, dataset, _, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        
        """
            Intrinsic information calculations for a given attribute
        """
        
        ### labels not used so made it _ for efficiency
        
        # how much data goes to each split, and length is number of splits
        _, splitCounts = np.unique(dataset[:, attribute], return_counts = True)
        
        
        intI = 0.0
        # looping through each split
        for splitCount in splitCounts:
            
            # finding the proportion of this split
            prop = splitCount/len(dataset)
            
            # multiplying proportion with its log
            intI += prop * np.log2(prop)
        
        intI = -intI
        
        return intI
    
    
    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        
        intI = self.calculate_intrinsic_information__(dataset, labels, attribute)
        
        # adding this check to avoid dividing by zero
        if (intI):
            return self.calculate_information_gain__(dataset, labels, attribute)/intI
        
        return 0


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """
        
        ### if all data belongs to the same class
        if (len(set(labels)) == 1):
            return TreeLeafNode(dataset, labels[0])
        
        
        ### else, we have to divide the tree further
        
        
        # finding the attributes that we can split further
        unused_attributes = list(set(range(len(self.features))) - set(used_attributes))
        
        
        # if there are no attributes left to test, we need to make leaf node with majority
        if (unused_attributes == []):
            vals, counts = np.unique(labels, return_counts = True)
            return TreeLeafNode(dataset, vals[np.argmax(counts)])
        
        
        # defining the gain function
        if self.criterion == "information gain":
            gainFun = self.calculate_information_gain__
            
        else: # self.criterion == "gain ratio"
            gainFun = self.calculate_gain_ratio__
        
        
        # finding the best gain and its attribute
        bestGain = -1.0
        bestAttr = -1
        
        # looping over each attribute and comparing its gain with the best gain
        for attr in unused_attributes:
            
            currGain = gainFun(dataset, labels, attr)
            
            if currGain > bestGain:
                bestGain = currGain
                bestAttr = attr
        
        
        # we have found the best attribute, now we add it to used_attributes
        used_attributes.append(bestAttr)
        print(f"Selected feature: {self.features[bestAttr]}")
        
        # creating the tree node for this attribute
        root = TreeNode(bestAttr)
        
        # finding the branches for the attribute
        branches = set(dataset[:, bestAttr])
        
        # looping for each branch, recursively calling ID3 in this loop
        for branch in branches:
            
            # finding the data and labels for this branch
            branchIndices = np.where(branch == dataset[:, bestAttr])[0]
            branchData = dataset[branchIndices]
            branchLabels = labels[branchIndices]
            
            # recursive call on ID3
            root.subtrees[branch] = self.ID3__(branchData, branchLabels, used_attributes)
        
        
        # return the tree after creating its branches
        return root
        
        
        
        
    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        """
            Your implementation
        """
        
        curr = self.root
        
        # iterating over the tree
        while isinstance(curr, TreeNode):
            curr = curr.subtrees[x[curr.attribute]]

        return curr.labels

    def train(self): # I CONVERTED INPUT INTO NP ARRAYS BECAUSE I AM USING NUMPY LIST FEATURES
        print(f"Training with criterion: {self.criterion}\n")
        self.root = self.ID3__(np.array(self.dataset), np.array(self.labels), [])
        print("\nTraining completed")