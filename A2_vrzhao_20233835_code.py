import scipy.io as spio
from datetime import datetime
import pandas as pd

# load in the data in a dictionary format
test_data_dict = spio.loadmat('test_images.mat', squeeze_me=True)
test_labels_dict = spio.loadmat('test_labels.mat', squeeze_me=True)
training_data_dict = spio.loadmat('train_images.mat', squeeze_me=True)
training_labels_dict = spio.loadmat('train_labels.mat', squeeze_me=True)

# converts the data into a np.array
test_data = test_data_dict['test_images']
test_labels = test_labels_dict['test_labels']
training_data = training_data_dict['train_images']
training_labels = training_labels_dict['train_labels']


# decision tree classifier
from sklearn.tree import DecisionTreeClassifier
def decisionTree(criterion, depth, data, labels):
    startTime = datetime.now()
    # generates a decision tree classifier with a random state of 0 based on the given criterion an max depth
    dT_classifier = DecisionTreeClassifier(random_state=0, criterion=criterion, max_depth=depth)
    # fit the classifier based on given data and labels
    dT_classifier.fit(data, labels)

    time =  datetime.now() - startTime

    # return the classifier and the training time
    return dT_classifier, time

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
def KNN(data, labels):
    startTime = datetime.now()
    # generates a KNN classifier with a n_neighbors of 3
    knn_classifier = KNeighborsClassifier(n_neighbors = 3)
    # fit the classifier based on given data and labels
    knn_classifier.fit(data, labels)

    time = datetime.now() - startTime

    # return the classifier and the training time
    return knn_classifier, time

# SVM classifier
from sklearn.svm import SVC
def SVM(data, labels):
    startTime = datetime.now()
    # generates a SVM classifier
    svm_classifier = SVC()
    # fit the classifier based on given data and labels
    svm_classifier.fit(data, labels)

    time = datetime.now() - startTime

    # return the classifier and the training time
    return svm_classifier, time

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
def randomForest(data, labels):
    startTime = datetime.now()
    # generates a random forest classifier with a random state of 0
    rf_classifier = RandomForestClassifier(random_state=0)
    # fit the classifier based on given data and labels
    rf_classifier.fit(data, labels)

    time = datetime.now() - startTime

    # return the classifier and the training time
    return rf_classifier, time

# Multilayer Perception classifier
from sklearn.neural_network import MLPClassifier
def MLP(layers, data, labels):
    startTime = datetime.now()
    # generate a multilayer perception with the given hidden layers
    mlp_classifier = MLPClassifier(hidden_layer_sizes=layers)
    # fit the classifier based on given data and labels
    mlp_classifier.fit(data, labels)

    time = datetime.now() - startTime

    # return the classifier and the training time
    return mlp_classifier, time

# function for merging and splitting the data to create a new set of training and test data/labels
from random import *
import numpy as np
def dataSplit():
    # merge test/train images and labels into new array
    data_set = np.append(training_data, test_data,axis = 0)
    label_set = np.append(training_labels, test_labels)
    for i in range(0,1000,1):
        # pick 1000 test/train images and labels without replacement into new array
        # pick a random position in the merged data/label set
        x = randint(0,11000-i)

        # creates a new array to hold the new test data/labels
        if (i == 0):
            test_data_split = [data_set[x]]
            test_labels_split = label_set[x]

        # appends selected data/labels to the previously created array
        else:
            test_data_split = np.concatenate((test_data_split,[data_set[x]]),axis = 0)
            test_labels_split = np.append(test_labels_split, label_set[x])

        # deletes selected data from the merged data/label sets
        data_set = np.delete(data_set,x,axis=0)
        label_set = np.delete(label_set,x)

    # returns the newly created training/test data/label sets
    return test_data_split, test_labels_split, data_set, label_set


# function for calculating the average precision
def precision(prediction, true_values):
    # generates two lists for contain the true positive and false positives
    TP = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]
    FP = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]


    for i in range (0,10,1):
        for j in range(0, len(prediction), 1):
            # checks for a specific prediction
            if(prediction[j] == i):
                # Checks to see if the prediction equals the true value
                # if true, than increment the true positive count
                # else increment the false positive count
                if(prediction[j] == true_values[j]):
                    TP[i] += 1
                else:
                    FP[i] += 1

    # generates an array to hold the individual precision values
    precision_1 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]

    # calculates the precision.
    for i in range (0,10,1):
        # if the denominator is 0, avoid a division by 0 error and set the precision to 0
        if (TP[i] + FP[i] == 0):
            precision_1[i] = 0.0
        else:
            precision_1[i] = TP[i] / (TP[i] + FP[i])

    # return the average precision
    return sum(precision_1)/len(precision_1)

# function for calculating the average recall
def recall(prediction, true_values):
    # generates two lists for contain the true positive and false negatives
    TP = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]
    FN = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]

    for i in range (0,10,1):
        for j in range(0, len(prediction), 1):
            # checks for a specific true value
            if(true_values[j] == i):
                # Checks to see if the prediction equals the true value
                # if true, than increment the true positive count
                # else increment the false negative count
                if(prediction[j] == true_values[j]):
                    TP[i] += 1
                else:
                    FN[i] += 1

    # generates an array to hold the individual recall values
    recall_1 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]

    x = 0
    # calculates the recall.
    for i in range (0,10,1):
        # if the denominator is 0, avoid a division by 0 error and set the recall to 0
        if (TP[i] + FN[i] == 0):
            recall_1[i] ==0.0
            x += 1
        else:
            recall_1[i] = TP[i] / (TP[i] + FN[i])

    # return the average recall, ommit values where the denominator was 0
    return sum(recall_1)/(len(recall_1) - x)

# function for calculating the f1 score
def f1(precision, recall):
    # return the f1 score
    return (2*precision*recall)/(precision+recall)

if __name__ == '__main__':
    tup1 = ("Classifier", "precision", "accuracy", "f1 score", "recall", "training time")

    # create a decision tree classifer with a creterion of gini and max depth of 5
    dT_classifier, time = decisionTree("gini",5, training_data, training_labels)
    prediction = dT_classifier.predict(test_data)
    precision_score = precision(prediction,test_labels)
    recall_score = recall(prediction,test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy =  dT_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 2
    tup2 = ("Decision Tree: Gini, Depth: 5", precision_score, acurracy, f1_score, recall_score, str(time))

    # create a decision tree classifer with a creterion of gini and max depth of 10
    dT_classifier, time = decisionTree("gini",10, training_data, training_labels)
    prediction = dT_classifier.predict(test_data)
    precision_score = precision(prediction,test_labels)
    recall_score = recall(prediction,test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy =  dT_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 3
    tup3 = ("Decision Tree: Gini, Depth: 10", precision_score, acurracy, f1_score, recall_score, str(time))

    # create a decision tree classifer with a creterion of entropy and max depth of 5
    dT_classifier, time = decisionTree("entropy",5, training_data, training_labels)
    prediction = dT_classifier.predict(test_data)
    precision_score = precision(prediction,test_labels)
    recall_score = recall(prediction,test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy =  dT_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 4
    tup4 = ("Decision Tree: Entropy, Depth: 5", precision_score, acurracy, f1_score, recall_score, str(time))

    # create a decision tree classifer with a creterion of entropy and max depth of 10
    dT_classifier, time = decisionTree("entropy",10, training_data, training_labels)
    prediction = dT_classifier.predict(test_data)
    precision_score = precision(prediction,test_labels)
    recall_score = recall(prediction,test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy =  dT_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 5
    tup5 = ("Decision Tree: Entropy, Depth: 10", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a KNN classifier
    knn_classifier, time = KNN(training_data, training_labels)
    prediction = knn_classifier.predict(test_data)
    precision_score = precision(prediction, test_labels)
    recall_score = recall(prediction, test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy = knn_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 6
    tup6 = ("KNN", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a SVM classifier
    svm_classifier, time = SVM(training_data, training_labels)
    prediction = svm_classifier.predict(test_data)
    precision_score = precision(prediction, test_labels)
    recall_score = recall(prediction, test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy = svm_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 7
    tup7 = ("SVM", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a random forest classifier
    rf_classifier, time = randomForest(training_data, training_labels)
    prediction = rf_classifier.predict(test_data)
    precision_score = precision(prediction, test_labels)
    recall_score = recall(prediction, test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy = rf_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 8
    tup8 = ("Random Forest", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a multilayer perception classifier with 1 hidden layer with 50 neurons
    mlp_classifier, time = MLP((50,),training_data, training_labels)
    prediction = mlp_classifier.predict(test_data)
    precision_score = precision(prediction, test_labels)
    recall_score = recall(prediction, test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy = mlp_classifier.score(test_data,test_labels)
    # saves the classifier results to tuple 9
    tup9 = ("Multilayer perception, Neurons: 50", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a multilayer perception classifier with 1 hidden layer with 100 neurons
    mlp_classifier, time = MLP((100,), training_data, training_labels)
    prediction = mlp_classifier.predict(test_data)
    precision_score = precision(prediction, test_labels)
    recall_score = recall(prediction, test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy = mlp_classifier.score(test_data, test_labels)
    # saves the classifier results to tuple 10
    tup10 = ("Multilayer perception, Neurons: 100", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a multilayer perception classifier with 2 hidden layer with 100 neurons in the first layer and 10 in the second
    mlp_classifier, time = MLP((100,10), training_data, training_labels)
    prediction = mlp_classifier.predict(test_data)
    precision_score = precision(prediction, test_labels)
    recall_score = recall(prediction, test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy = mlp_classifier.score(test_data, test_labels)
    # saves the classifier results to tuple 11
    tup11 = ("Multilayer perception, Neurons: 100, 10", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a multilayer perception classifier with 2 hidden layer with 50 neurons in the first layer and 20 in the second
    mlp_classifier, time = MLP((50,20), training_data, training_labels)
    prediction = mlp_classifier.predict(test_data)
    precision_score = precision(prediction, test_labels)
    recall_score = recall(prediction, test_labels)
    f1_score = f1(precision_score, recall_score)
    acurracy = mlp_classifier.score(test_data, test_labels)
    # saves the classifier results to tuple 12
    tup12 = ("Multilayer perception, Neurons: 50, 20", precision_score, acurracy, f1_score, recall_score, str(time))

    # generates a new set of test/training data/labels using the data split function
    test_data_split, test_labels_split, training_data_split, training_labels_split = dataSplit()

    # creates a KNN classifier using the newly created test/training data/label sets
    knn_classifier, time = KNN(training_data_split,training_labels_split)
    prediction = knn_classifier.predict(test_data_split)
    precision_score = precision(prediction, test_labels_split)
    recall_score = recall(prediction, test_labels_split)
    f1_score = f1(precision_score, recall_score)
    acurracy = knn_classifier.score(test_data_split,test_labels_split)
    # saves the classifier results to tuple 13
    tup13 = ("KNN_dataSplit", precision_score, acurracy, f1_score, recall_score, str(time))

    # creates a SVM classifier using the newly created test/training data/label sets
    svm_classifier, time = SVM(training_data_split,training_labels_split)
    prediction = svm_classifier.predict(test_data_split)
    precision_score = precision(prediction, test_labels_split)
    recall_score = recall(prediction, test_labels_split)
    f1_score = f1(precision_score, recall_score)
    acurracy = svm_classifier.score(test_data_split,test_labels_split)
    # saves the classifier results to tuple 14
    tup14 = ("SVM_dataSplit", precision_score, acurracy, f1_score, recall_score, str(time))

    # puts all the saved data into a single summary table and prints it
    summary_table = [tup1,tup2,tup3,tup4,tup5,tup6, tup7, tup8, tup9, tup10, tup11, tup12, tup13, tup14]
    print summary_table

    # writes results to a csv
    import csv
    with open('summary_table.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
        for row in summary_table:
            wr.writerow(row)