from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

#get training and testing data
data = pd.read_csv('../bioassay/AID373red_train.csv')
test_data = pd.read_csv('../bioassay/AID373red_test.csv')

data["Outcome"] = data["Outcome"].map({"Inactive": 0, "Active": 1})
test_data["Outcome"] = test_data["Outcome"].map({"Inactive": 0, "Active": 1})

x_train = data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_train = data["Outcome"]

x_test = test_data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_test = test_data["Outcome"]

#build xgboost model
clf = XGBClassifier()  
clf.fit(x_train , y_train)  

#testing metrics
y_prediction = clf.predict(x_test)  
predictions = [round(value) for value in y_prediction]  

accuracy = accuracy_score(y_test, predictions)  
print("accuracy: ", accuracy)
print("f1 score: ", f1_score(y_test, predictions)) 