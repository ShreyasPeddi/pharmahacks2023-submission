Winner, 1st place

Based on the research paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2820499/

Drug screening is a necessary process that leads to new drug discovery. However, it takes an average of 15 years and $800 million to bring a drug to the market. With the advent of ML, these problems are eliminated as ML models can be used to predict interactions between ligands and protein-of-interest. Consequently, we sought to improve upon existing drug-screening ML models. In our work (Challenge 1), to understand which descriptors correlated to the outcome, we used Binary Logistic Regression to pre-process our data because it could assess multiple independent variables at a time to see how they were correlated to the Boolean outcome (Active or Inactive). Descriptors which resulted in a p-value less than 0.05 after the analysis were used for modelling. To build our models, we used Python and modules such as pandas, matplotlib, seaborn, statsmodels, scikit-learn. We used a combination of ML models including Random Forest with Randomized Search CV, XGBoost, SVM, and Naive Bayes with kernel density. We combined them in a Stacked model with XGBoost as the strongest learner. Through our model, we monitored accuracy, f1 and cohen-kappa scores. We achieved an accuracy of 0.98 and average f1 score of 0.66 in predicting protein-ligand interactions.

We built our model using Python and several libraries:

- pandas
- matplotlib
- seaborn
- statsmodels
- scikit-learn

We used 4 ML models: Random Forest with RandomizedSearchCV, XGBoost, SVM, and Naive Bayes with kernel density.
Our rationale for using our chosen models is mentioned below:

- Random Forest - We believed a decision tree algorithm would be ideal. More importantly, the researchers in the paper ran out of memory space since they used only 2 gigabytes of heap space for Windows system. Since we have computers with faster processing, we were able to implement it. We also used RandomizedSearchCV to pinpoint what max_depth and n_estimators are best for our dataset.

- XGBoost (gaussian) - It has both linear model solver and tree learning algorithms.

- Support vector machine - It worked well, as mentioned in the research paper.

- Gaussian Naive Bayes with kernel density estimation - It worked moderately well for researchers, as mentioned in the paper we were given, but we implemented it with kernel density estimation to optimize it.

We also used a logistic regression model for preprocessing.
