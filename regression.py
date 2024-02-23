import pandas as pd
import matplotlib.pyplot as plt

x1x2y = pd.read_excel("X1X2Y.xlsx")

y = x1x2y['y']
x1x2 = x1x2y.drop(['y'], axis = 1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x1x2, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier
x1x2y_dt_classifier = DecisionTreeClassifier(criterion='entropy')
x1x2y_dt_classifier.fit(x_train, y_train)

y_pred = x1x2y_dt_classifier.predict(x_test)

from sklearn.inspection import DecisionBoundaryDisplay

dt_decbound = DecisionBoundaryDisplay.from_estimator(x1x2y_dt_classifier, x_train, response_method="predict", xlabel='Decision Tree', ylabel=' ', alpha=0.5,)
dt_decbound.ax_.scatter(x_train.values[:, 0], x_train.values[:, 1], c = y_train, edgecolor="k")
plt.show()

print('Accuracy Score for DecisionTree: ', accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
x1x2y_l_regressor = LogisticRegression(random_state=0)
x1x2y_l_regressor.fit(x_train, y_train)

y_pred = x1x2y_l_regressor.predict(x_test)

lr_decbound = DecisionBoundaryDisplay.from_estimator(x1x2y_l_regressor, x_train, response_method="predict", xlabel='Logistic Regressor', ylabel=' ', alpha=0.5,)
lr_decbound.ax_.scatter(x_train.values[:, 0], x_train.values[:, 1], c = y_train, edgecolor="k")
plt.show()

print('Accuracy Score for LogisticRegression: ', accuracy_score(y_test, y_pred))

from sklearn.svm import SVC
x1x2y_svmlinear_classifier = SVC(kernel='linear')
x1x2y_svmlinear_classifier.fit(x_train, y_train)

y_pred = x1x2y_svmlinear_classifier.predict(x_test)

lsvm_decbound = DecisionBoundaryDisplay.from_estimator(x1x2y_svmlinear_classifier, x_train, response_method="predict", xlabel='Linear SVM', ylabel=' ', alpha=0.5,)
lsvm_decbound.ax_.scatter(x_train.values[:, 0], x_train.values[:, 1], c = y_train, edgecolor="k")
plt.show()

print('Accuracy Score for LinearSVM: ', accuracy_score(y_test, y_pred))

x1x2y_svmrbf_classifier = SVC(kernel='rbf')
x1x2y_svmrbf_classifier.fit(x_train, y_train)

y_pred = x1x2y_svmrbf_classifier.predict(x_test)

rbfsvm_decbound = DecisionBoundaryDisplay.from_estimator(x1x2y_svmrbf_classifier, x_train, response_method="predict", xlabel='RBF SVM', ylabel=' ', alpha=0.5,)
rbfsvm_decbound.ax_.scatter(x_train.values[:, 0], x_train.values[:, 1], c = y_train, edgecolor="k")
plt.show()

print('Accuracy Score for RBFSVM: ', accuracy_score(y_test, y_pred))