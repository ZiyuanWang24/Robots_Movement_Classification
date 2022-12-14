# Software components
We create a jupyter notebook file, Comparison_of_classifiers.ipynb, to manage and visualize the data. In the file, there are five different machine learning classification algorithms to analyze the ultrasounds data. The results are shown as confusion matrix and predication accuracy.
## src/Classifiers.py:
It is a python class file that includes the Linear Classification, Support Vector Machine, KNN Classification, Neutral Network, and Gaussian Naive Bayes.\
Input: Orignal data set in .csv format.\
Output: robots_movement_classifier class which is ready for training in different classification algorithm.
## robots_movement_classifier.knnclassifier_learning():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of KNN Classification algorithm.
## robots_movement_classifier.SVM_learning():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Support Vector Machine Classification algorithm.
## robots_movement_classifier.GaussianNB_learning():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Gaussian Naive Bayes Classification algorithm.
## robots_movement_classifier.Linear_Classifier():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Linear Classification algorithm.
## robots_movement_classifier.NeuralNet_Classifier():
Input: None\
Ouput: Confusion matrix, prediction accuracy, trained model of Neutral Network Classification algorithm.
# User Interface
<img src="img/MLComparison.png" height="400" width="400" align=right></img>
## Example
Through our testing with the data we use, KNN Classification got the highest accuracy of 98% in all three kinds of datasets. The following figures shows the comparison for all five machine learning algorithms. We can see clearly the different of the accuracy rate of these 5 algorithms. Another figure shows the accuracy rate and confusion matrix of all 5 Machine Learning Algorithm.
<img src="img/5MLConfusionMX.png" height="200" width="1000" align=center></img>

# Preliminary plan
