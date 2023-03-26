# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns               #for statistical plotting
import statsmodels.api as sm        #to fit our logistic regression model
from statsmodels.formula.api import logit

data = pd.read_csv('../bioassay/AID362red_train.csv')

# sns.regplot(x = "MW", y = "Outcome", 
#             y_jitter = 0.03, 
#             data = data, 
#             logistic = True,
#             ci = None)
# plt.show()

#coverting outcome data to integers
data["Outcome"] = data["Outcome"].map({"Inactive": 0, "Active": 1})
print(data.head())

#add logistic fit
formula = ('Outcome ~ WBN_GC_L_A + WBN_GC_H_A + WBN_GC_L_B + WBN_GC_H_B + WBN_GC_L_C + WBN_GC_H_C + WBN_GC_L_D + WBN_GC_H_D + WBN_EN_L_A + WBN_EN_H_A + WBN_EN_L_B + WBN_EN_H_B + WBN_EN_L_C + WBN_EN_H_C + WBN_EN_L_D + WBN_EN_H_D + WBN_LP_L_A + WBN_LP_H_A + WBN_LP_L_B + WBN_LP_H_B + WBN_LP_L_C + WBN_LP_H_C + WBN_LP_L_D + WBN_LP_H_D + XLogP + PSA + NumRot + NumHBA + NumHBD + MW + BBB + BadGroup')
model = logit(formula = formula, data = data).fit()
print(model.summary())

#generate ame
ame = model.get_margeff(at='overall', method='dydx')
print (ame.summary())