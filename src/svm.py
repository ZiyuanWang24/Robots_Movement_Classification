import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import datasets
from sklearn.svm import SVC


def SVM_learning(datasets):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(datasets, test_size=0.2)
    xtrain = train.iloc[:, 0:4].values
    ytrain = train.iloc[: , -1].values
    xtest  = test.iloc[:, 0:4].values
    ytest  = test.iloc[:,-1].values
    classifier = SVC(kernel='rbf', random_state = 1)
    classifier.fit(xtrain,ytrain)
    y_pred_test = classifier.predict(xtest)
    y_pred_train = classifier.predict(xtrain)
    return y_pred_test
