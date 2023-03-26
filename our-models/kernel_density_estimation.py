#imports
import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

#build naive bayes model with kernel density estimation

# #radial kernel
# nbmodel1 = NaiveBayesClassifier(bandwidth=1, kernel='radial')
# nbmodel1.fit(x_train, y_train)
# print("radial kernel with bandwidth 1: ", nbmodel1.score(x_test, y_test))

# #hypercube kernel
# nbmodel2 = NaiveBayesClassifier(bandwidth=1, kernel='hypercube')
# nbmodel2.fit(x_train, y_train)
# print("hypercube kernel with bandwidth 1: ", nbmodel2.score(x_test, y_test))

clf = GaussianNB()
clf.fit(x_train, y_train)

#training metrics
prediction = clf.predict(x_test)
print("accuracy: ", accuracy_score(y_test, prediction))
print("f1 score: ", f1_score(y_test,prediction))