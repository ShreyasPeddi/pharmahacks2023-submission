#imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

#get training and testing data
data = pd.read_csv('./bioassay/AID373red_train.csv')
test_data = pd.read_csv('./bioassay/AID373red_test.csv')

data["Outcome"] = data["Outcome"].map({"Inactive": 0, "Active": 1})
test_data["Outcome"] = test_data["Outcome"].map({"Inactive": 0, "Active": 1})

x_train = data.drop(["Outcome"], axis=1)
y_train = data["Outcome"]

x_test = test_data.drop(["Outcome"], axis=1)
y_test = test_data["Outcome"]

#random forest model
RF_classifier = RandomForestClassifier(max_depth=6, n_estimators = 178)

#xgboost model
XGB_classifier = XGBClassifier()

#svm model
SVM_classifier = SVC(kernel='rbf')

#naive bayes model
NB_classifier = GaussianNB()

#stack all models
estimators = [RF_classifier, SVM_classifier, NB_classifier]
estimators = [
     ('rf', RandomForestClassifier(max_depth=6, n_estimators = 178)),
     ('svm', SVC(kernel='rbf')),
     ('nb', GaussianNB())
]

aggregate_model = StackingClassifier(estimators = estimators, 
                                    final_estimator = XGB_classifier)

print("score: ", aggregate_model.fit(x_train, y_train).score(x_test, y_test))

#testing metrics
y_predictions = aggregate_model.predict(x_test)

f1 = f1_score(y_test, y_predictions, average="weighted", zero_division=0)
cohen_kappa = cohen_kappa_score(y_test, y_predictions)
accuracy = accuracy_score(y_test, y_predictions)  

print("accuracy: ", accuracy)
print("f1 score: ", f1)
print("cohen kappa score: ", cohen_kappa)
print(classification_report(y_test, y_predictions))
