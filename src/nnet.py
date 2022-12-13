#!/usr/bin/env python
# coding: utf-8

# In[1]:


def NNetclassifier(dataset):
    X = datasets.iloc[:, [0,1,2,3]].values
    y = datasets.iloc[:, -1].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred
def Linclassifier(dataset):
    X = datasets.iloc[:, [0,1,2,3]].values
    y = datasets.iloc[:, -1].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    from sklearn import linear_model
    classifier = linear_model.LogisticRegression(penalty ='l2',max_iter=500,multi_class= 'ovr')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred


# In[ ]:




