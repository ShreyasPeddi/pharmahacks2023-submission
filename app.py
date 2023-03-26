#imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

#get training and testing data
data = pd.read_csv('./bioassay/AID373red_train.csv')
test_data = pd.read_csv('./bioassay/AID373red_test.csv')

data["Outcome"] = data["Outcome"].map({"Inactive": 0, "Active": 1})
test_data["Outcome"] = test_data["Outcome"].map({"Inactive": 0, "Active": 1})

x_train = data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_train = data["Outcome"]

x_test = test_data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_test = test_data["Outcome"]

#random forest model
RF_classifier = RandomForestClassifier(max_depth=6, n_estimators = 178)
RF_classifier.fit(x_train, y_train)

#xgboost model
XGB_classifier = XGBClassifier()
XGB_classifier.fit(x_train , y_train)  

#svm model
SVM_classifier = SVC(kernel='rbf')
SVM_classifier.fit(x_train, y_train)

#naive bayes model
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)

#stack all models
estimators = [RF_classifier, XGB_classifier, SVM_classifier, NB_classifier]
estimators = [
     ('rf', RandomForestClassifier(max_depth=6, n_estimators = 178)),
     ('xgb', XGBClassifier()),
     ('svm', SVC(kernel='rbf')),
     ('nb', GaussianNB())
]

aggregate_model = StackingClassifier(estimators = estimators, 
                                    final_estimator = RF_classifier)

print(aggregate_model.fit(x_train, y_train).score(x_test, y_test))

#testing metrics
y_predictions = aggregate_model.predict(x_test)

f1 = f1_score(y_test, y_predictions, average="micro")
cohen_kappa = cohen_kappa_score(y_test, y_predictions)

print(f1)
print(cohen_kappa)
print(classification_report(y_test, y_predictions))
