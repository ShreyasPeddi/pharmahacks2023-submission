#imports
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

#get training and testing data
data = pd.read_csv('../bioassay/AID373red_train.csv')
test_data = pd.read_csv('../bioassay/AID373red_test.csv')

data["Outcome"] = data["Outcome"].map({"Inactive": 0, "Active": 1})
test_data["Outcome"] = test_data["Outcome"].map({"Inactive": 0, "Active": 1})

x_train = data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_train = data["Outcome"]

x_test = test_data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_test = test_data["Outcome"]

#build svm model with gaussian kernel
svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)

#training metrics
y_prediction = svclassifier.predict(x_test)
print(confusion_matrix(y_test, y_prediction))
print(classification_report(y_test,y_prediction))