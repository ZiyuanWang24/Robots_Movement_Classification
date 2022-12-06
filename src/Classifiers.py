import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class robots_movement_classifier():

    def __init__(self, datasets):
        self.X = datasets.iloc[:, [0,1,2,3]].values
        self.y = datasets.iloc[:, -1].values
        le = LabelEncoder()
        self.X[:,0] = le.fit_transform(self.X[:,0])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size = 0.20, random_state = 0)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def knnclassifier_learning(self):        
        self.knnclassifier = KNeighborsClassifier(n_neighbors = 8, p = 2)
        self.knnclassifier.fit(self.X_train, self.y_train)
        y_pred = self.knnclassifier.predict(self.X_test)
        self.knn_cm = confusion_matrix(self.y_test, y_pred)
        self.knn_ac = accuracy_score(self.y_test,y_pred)
        # print("knn accuracy = ", ac)
        return self.knn_cm , self.knn_ac, self.knnclassifier


    def SVM_learning(self):
        #data = data.replace('Move-Forward',1)
        #data = data.replace('Slight-Right-Turn',2)
        #data = data.replace('Slight-Left-Turn',3)
        #data = data.replace('Sharp-Right-Turn',4)
        #from sklearn.model_selection import train_test_split
        #train, test = train_test_split(data, test_size=0.2)
        #xtrain = train.iloc[:, 0:24].values
        #ytrain = train.iloc[: , -1].values
        #xtest  = test.iloc[:, 0:24].values
        #ytest  = test.iloc[:,-1].values

        self.SVMclassifier = SVC(kernel='rbf', random_state = 1)
        self.SVMclassifier.fit(self.X_train,self.y_train)
        y_pred_test = self.SVMclassifier.predict(self.X_test)
        self.SVM_cm = confusion_matrix(self.y_test,y_pred_test)
        self.SVM_ac = float(self.SVM_cm.diagonal().sum())/len(self.y_test)
        # print("\nAccuracy Of SVM For The Given Dataset : ", self.SVM_ac)
        return self.SVM_cm, self.SVM_ac, self.SVMclassifier

    def GaussianNB_learning(self):
        #data = data.replace('Move-Forward',1)
        #data = data.replace('Slight-Right-Turn',2)
        #data = data.replace('Slight-Left-Turn',3)
        #data = data.replace('Sharp-Right-Turn',4)
        #from sklearn.model_selection import train_test_split
        #train, test = train_test_split(data, test_size=0.2)
        #xtrain = train.iloc[:, 0:24].values
        #ytrain = train.iloc[: , -1].values
        #xtest  = test.iloc[:, 0:24].values
        #ytest  = test.iloc[:,-1].values
        self.GNBclassifier = GaussianNB()
        self.GNBclassifier.fit(self.X_train, self.y_train)
        y_pred_test = self.GNBclassifier.predict(self.X_test)
        self.GNB_cm = confusion_matrix(self.y_test,y_pred_test)
        self.GNB_ac = float(self.GNB_cm.diagonal().sum())/len(self.y_test)
        # print("\nAccuracy Of SVM For The Given Dataset : ", self.GNB_ac)
        return self.GNB_cm, self.GNB_ac, self.GNBclassifier
    def Linear_Classifier(self):
        self.Linclassifier = linear_model.LogisticRegression(penalty ='l2',max_iter=500,multi_class= 'ovr')
        self.Linclassifier.fit(self.X_train, self.y_train)
        y_pred_test = self.Linclassifier.predict(self.X_test)
        self.Lin_cm = confusion_matrix(self.y_test,y_pred_test)
        self.LIn_ac = float(self.Lin_cm.diagonal().sum())/len(self.y_test)
        return self.Lin_cm, self.Lin_ac, self.Linclassifier
    def NeuralNet_Classifier(self):
        self.NNetclassifier = MLPClassifier()
        self.NNetclassifier.fit(self.X_train, self.y_train)
        y_pred_test = self.NNetclassifier.predict(self.X_test)
        self.NNet_cm = confusion_matrix(self.y_test,y_pred_test)
        self.NNet_ac = float(self.NNet_cm.diagonal().sum())/len(self.y_test)
        return self.NNet_cm, self.NNet_ac, self.NNetclassifier

