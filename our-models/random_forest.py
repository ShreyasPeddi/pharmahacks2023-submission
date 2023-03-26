#imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from scipy.stats import randint

#get training and testing data
data = pd.read_csv('../bioassay/AID373red_train.csv')
test_data = pd.read_csv('../bioassay/AID373red_test.csv')

data["Outcome"] = data["Outcome"].map({"Inactive": 0, "Active": 1})
test_data["Outcome"] = test_data["Outcome"].map({"Inactive": 0, "Active": 1})

x_train = data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_train = data["Outcome"]

x_test = test_data[["WBN_GC_L_A","WBN_GC_H_A","WBN_GC_L_B","WBN_GC_H_B","WBN_GC_L_C","WBN_GC_H_C","WBN_GC_L_D","WBN_GC_H_D","WBN_EN_L_A","WBN_EN_H_A","WBN_EN_L_B","WBN_EN_H_B","WBN_EN_L_C","WBN_EN_H_C","WBN_EN_L_D","WBN_EN_H_D","WBN_LP_L_A","WBN_LP_H_A","WBN_LP_L_B","WBN_LP_H_B","WBN_LP_L_C","WBN_LP_H_C","WBN_LP_L_D","WBN_LP_H_D","XLogP","PSA","NumRot","NumHBA","NumHBD","MW","BBB","BadGroup"]]
y_test = test_data["Outcome"]

#build random forest model with hyperparameter tuning
params = {'n_estimators': randint(50, 500),
              'max_depth': randint(1, 200)}

clf = RandomForestClassifier()

random_search = RandomizedSearchCV(clf, 
                                 param_distributions = params, 
                                 n_iter = 10, 
                                 cv = 10)

random_search.fit(x_train, y_train)

#best paramters for our model
best_clf = random_search.best_estimator_
print("Best hyperparameters: ",  random_search.best_params_)

#testing predictions
prediction = best_clf.predict(x_test)

#testing metrics
print("confusion matrix: ", confusion_matrix(y_test, prediction))
print("accuracy: ", accuracy_score(y_test, prediction))
print("f1 score: ", f1_score(y_test,prediction))