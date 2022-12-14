# Software components
We create a jupyter notebook file, Comparison_of_classifiers.ipynb, to manage and visualize the data. In the file, there are five different machine learning classification algorithms to analyze the ultrasounds data. The results are shown as confusion matrix and predication accuracy.
## src/Classifiers.py:
It is a python class file that includes the Linear Classification, Support Vector Machine, KNN Classification, Neutral Network, and Gaussian Naive Bayes.\
Input: Orignal data set in .csv format.\
Output: robots_movement_classifier class which is ready for training in different classification algorithm.\
## robots_movement_classifier.knnclassifier_learning():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of KNN Classification algorithm.\
## robots_movement_classifier.SVM_learning():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Support Vector Machine Classification algorithm.\
## robots_movement_classifier.GaussianNB_learning():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Gaussian Naive Bayes Classification algorithm.\
## robots_movement_classifier.Linear_Classifier():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Linear Classification algorithm.\
## robots_movement_classifier.NeuralNet_Classifier():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Neutral Network Classification algorithm.\


High level description of the software components such as: data manager, which provides a simplified interface to your data and provides application specific features (e.g., querying data subsets); and visualization manager, which displays data frames as a plot. Describe at least 3 components specifying: what it does, inputs it requires, and outputs it provides.
Interactions to accomplish use cases. Describe how the above software components interact to accomplish at least one of your use cases.
Preliminary plan. A list of tasks in priority order.
