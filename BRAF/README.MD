This project aims to create a Biased Random Forest Algorithm according to Bader-El-Den, Teitei, & Perry 2018


This project was done to create the algorithm, not necessarily to optimise it for this particular dataset.  With that said, there is still plenty of feature engineering, data wrangling, and model parameter tuning that can be done.


The dataset used for this project was the Pima Diabetes dataset, which can be found on Kaggle.com


Random_Forest_module.py --> this file contains all the classes and functions needed to implement the BRAF alogrithm

Pima EDA.ipynb --> this notebook contains the exploratory data analysis upon which certain data processes were chosen

Pima BRAF Analysis.ipynb --> this notebook contains the implementation of the BRAF algorithm with 10-fold cross validation and a test set.  It prints out the precision, recall, AUROC, and AUPRC for both the cross validation and test set.  It also creates PRC and ROC plots for both the cross validation and test set, which are also in this folder.


These were used as resources for building the RF algorithm:
http://course18.fast.ai/lessonsml1/lesson7.html
https://machinelearningmastery.com/implement-random-forest-scratch-python/

This was used as reference for getting K folds for the cross validation:
https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/model_selection/_split.py#L49

This was used as a reference for the ROC/PRC plots:
https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python