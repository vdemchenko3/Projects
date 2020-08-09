import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error,median_absolute_error,r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import patsy
import pickle
import sys
sys.path.append('../scripts')
import function

merged_df = pd.read_csv("../Clean_data/Merged_data.csv")

# change column names for convenience
merged_df.rename(columns={'Total Conversions':'Total_Conversions',
                          'Site Bin':'Sb',
                          'Placement Pixel Size Bin':'Pb'},
                 inplace=True)

# remove some outliers
merged_df = merged_df[np.log10(merged_df['Total_Conversions']+1) < 5.0]

# variables and formula for training the model
num_variables = ['Reach']
cat_variables = ['Sb','Pb']
formula = 'Total_Conversions ~ Reach + Reach:Sb + Reach:Pb + Reach:Sb:Pb - 1'

#condition for SMOTE
smote_cond = np.log10(merged_df['Total_Conversions']+1)>1.0

#make transformations for numerical variables
merged_df['Reach'],_,_ = function.make_all_transform(merged_df,
                                                     'Reach',
                                                     log=True,
                                                     normalize=True)

merged_df['Total_Conversions'],_,_ = function.make_all_transform(merged_df,
                                                               'Total_Conversions',
                                                               log=True,
                                                               normalize=True)

# split between smote and test data and apply SMOTE
merged_vars = merged_df[num_variables + cat_variables]
merged_response = merged_df[['Total_Conversions']]
X_smote,X_test,Y_smote,Y_test = train_test_split(merged_vars,
                                                 merged_response,
                                                 test_size = 0.2,
                                                 random_state=12)
df_smote = pd.concat([X_smote,Y_smote],axis=1)
df_smote = function.apply_smote(df_smote, 
                                smote_cond, 
                                cat_vars=cat_variables,
                                one_hot=False)

# build one-hot encoded matrix with patsy
Y_train,X_train_encoded = patsy.dmatrices(formula,
                                          df_smote,
                                          return_type='dataframe')

#Model training
params = {'alpha_1':np.linspace(1e-6,1,10000),
          'alpha_2':np.linspace(1e-6,1,10000),
          'lambda_1':np.linspace(1e-6,1,10000),
          'lambda_2':np.linspace(1e-6,1,10000)}

reg = RandomizedSearchCV(BayesianRidge(fit_intercept = False),
                         param_distributions = params, 
                         cv = 5,
                         random_state = 123, 
                         n_iter = 1000)
reg.fit(X_train_encoded,Y_train.values.flatten())

# Get metrics form test set
build_encoding = X_train_encoded.design_info
X_test_encoded, = patsy.build_design_matrices([build_encoding], 
                                               X_test,
                                               return_type='dataframe')

y_pred = reg.predict(X_test_encoded)
y_true = Y_test.values.flatten()

rmse = np.sqrt(mean_squared_error(y_true,y_pred))
median_abs_error = median_absolute_error(y_true,y_pred)
mae = mean_absolute_error(y_true,y_pred)
r2 = r2_score(y_true,y_pred)

print("RMSE = {:.3f}".format(rmse))
print("Median Absolute Error = {:.3f}".format(median_abs_error))
print("Mean Absolute Error = {:.3f}".format(mae))
print("R2 = {:.3f}".format(r2))

#Save trained model as pickle
dict_to_pickle = {'model':reg.best_estimator_, 'formula':formula,
                  'train_data':df_smote}
pickle.dump(dict_to_pickle, open( "ReachSbPb.p", "wb" ))
