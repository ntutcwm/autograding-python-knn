# -*- coding: utf-8 -*-
# Reference: https://nycdatascience.com/blog/student-works/machine-learning/knn-classifier-from-scratch-numpy-only/

import pandas as pd
import numpy as np
import argparse

KNN = 2
TRAIN_FILE = 'IRIS.csv'
TEST_FILE = 'iris_test.csv'

# Calculate all Euclidean distances between training and test data

def knn_calc_dists(xTrain, xTest, k):
    """
    Finds the k nearest neighbors of xTest in xTrain.
    Input:
        xTrain = n x d matrix. n=rows and d=features
        xTest = m x d matrix. m=rows and d=features (same amount of features as xTrain)
        k = number of nearest neighbors to be found
    Output:
        dists = distances between xTrain/xTest points. Size of n x m
        indices = kxm matrix with indices of yTrain labels
    """
    indices = np.zeros([xTrain.shape[0], xTest.shape[0]])
    distances = np.zeros([xTrain.shape[0], xTest.shape[0]])
    return indices[0:k, : ], distances[0:k, : ]

def knn_predict(xTrain, yTrain, xTest, k=3):
    """
    Input:
        xTrain = n x d matrix. n=rows and d=features
        yTrain = n x 1 array. n=rows with label value
        xTest = m x d matrix. m=rows and d=features
        k = number of nearest neighbors to be found
    Output:
        predictions = predicted labels, ie preds(i) is the predicted label of xTest(i,:)
    """
    indices, distances = knn_calc_dists(xTrain, xTest, k)
    rows, columns = indices.shape
    predictions = []
    for j in range(columns):
        predictions.append(['N/A'])
    
    return np.array(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K Nearest Neighbor Classifier")
    parser.add_argument("--train-csv", help="Training data in CSV format. Labels are stored in the last column.", required=True)
    parser.add_argument("--test-csv", help="Test data in CSV format", required=True)
    parser.add_argument("--num_k", "-k", dest="K", help="Number of nearest neighbors", default=3, type=int)
    args = parser.parse_args()

    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv(args.train_csv)
    train_data = train_df.iloc[:,:-1].to_numpy()
    train_label = train_df.iloc[:,-1:].to_numpy() # Split labels in last column

    test_df = pd.read_csv(args.test_csv)
    test_data = test_df.iloc[:,:-1].to_numpy()
    test_label = test_df.iloc[:,-1:].to_numpy() # Split labels in last column

    predictions = knn_predict(train_data, train_label, test_data, args.K)

    # Save prediction results
    #np.savetxt("predictions.csv", predictions, delimiter=',') # Not working for strings
    df = pd.DataFrame(predictions)
    df.to_csv("predictions.csv", header=False, index=False)

    # Calculate accuracy
    result = predictions == test_label
    accuracy = sum(result == True) / len(result)
    print('Evaluate KNN(K=%d) on Iris Flower dataset. Accuracy = %.2f' % (args.K, accuracy))
