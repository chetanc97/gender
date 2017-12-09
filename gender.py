from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
x = np.array(X)
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
G =  [0, 0 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 0]
y= np.array(G)
# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_svm_linear = SVC(kernel='linear', C = 1.0)
clf_perceptron = Perceptron()
clf_KNN =  KNeighborsClassifier()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_svm_linear.fit(X , Y)
clf_KNN.fit(X,Y)

# Testing using the same data
J = [[183 ,79 ,42], [176, 72, 44], [165, 62, 39], [157, 54, 39], [161, 67, 42], [189, 95, 43], [178, 67, 34],
     [192, 73, 42], [145, 57, 38], [172, 77, 46], [180, 86, 45]]
pred_tree = clf_tree.predict(J)
print ("With decision tree :  ")
print(pred_tree)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(J)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(J)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(J)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

pred_svm_linear = clf_svm_linear.predict(J)
acc_svm_linear = accuracy_score(Y, pred_svm_linear) * 100
print('Accuracy for linear SVM: {}'.format(acc_svm_linear))


# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))


w = clf_svm_linear.coef_[0]
#print(w)

a = -w[0] / w[1]

xx = np.linspace(0,135)
yy = a * xx - clf_svm_linear.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
# print(x)
# print(y)
plt.scatter(x[:, 2], y, c = G)
plt.legend()
plt.show()
