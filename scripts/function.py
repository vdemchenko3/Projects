def scale_and_standardize(df):

    # scale and standardize a variable
    # input: pandas dataframe with columns to be modified
    # returns: pandas dataframe with scaled and standardized columns and with
    #          the same index scheme; mean and var values.

    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    scaler = StandardScaler()

    scaler.fit(df)
    transform = scaler.transform(df)

    new_df = pd.DataFrame(transform, columns=df.columns, index=df.index)

    return new_df,scaler.mean_,scaler.var_

def calculate_reach(df):

    # Calculate the Reach variable from clicks and impressions columns
    # Input:   pandas dataframe to add the Reach column to
    # Returns: pandas dataframe with added Reach column

    import numpy as np

    cond1 = df['Impressions'] == 0
    cond2 = df['Clicks'] > 0

    df['Reach'] = cond1 & cond2

    ix_true = df.loc[df['Reach'] == True,'Reach'].index
    ix_false = df.loc[df['Reach'] == False,'Reach'].index

    # good cases of impressions
    df_good = df[~(cond1 & cond2)]

    # use median click rate to impute 0 impressions on ill cases
    df_ill = df[cond1 & cond2]
    clicks = df_good.loc[df_good['Placement Pixel Size Bin'] == 'Tracking',
                            'Clicks']
    impressions = df_good.loc[df_good['Placement Pixel Size Bin'] ==' Tracking',
                            'Impressions']
    click_rates = clicks.div(impressions)

    median_click_rate = np.median(click_rates)

    df.loc[ix_true,'Reach'] = df.loc[ix_true,'Clicks']/median_click_rate
    df.loc[ix_false,'Reach'] = df.loc[ix_false,'Impressions']
    df['Reach'] = df['Reach'].astype(float)

    return df

def make_all_transform(df, col_name, log=False, normalize=False, scale=False):

	# Function to enable various transforms to be applied to columns of an input dataframe
	# Input: df: 		dataframe to transform
	#		 col_name: 	column name to transform
	#		 log:		Boolean, True to apply log10+1 transform
	#		 normalize: Boolean, True to normalize column by duration
	#		 scale: 	Boolean, True to scale and standardize - attempting to get a zero centered distribution
	# Output df[col_name] gets updated as per the three boolean inputs

    import pandas as pd
    import numpy as np

    duration = df['Duration']
    series = df[col_name]
    mean = 0
    var = 0

    if normalize:
        series = series.div(duration)
    if log:
        series = np.log10(series+1)
    if scale:
        series = pd.DataFrame(series, columns=[col_name])
        series,mean,var = scale_and_standardize(series)
        series = series[col_name]

    return series,mean,var

def apply_smote(df, index, cat_vars = False, one_hot = True):

    # Applies smote to the df dataframe in the Total Conversions variable.
    # X should not have any type pf transformations
    # Input: df: dataframe to be smoted.
    #        index: True or False, indicating which value is from the minority
    #               group (True) and which is from the majority group (False)
    # Returns: smoted and one-hot encoded

    from imblearn.over_sampling import SMOTENC,SMOTE
    import pandas as pd
    import numpy as np

    # assign class 1 to minorty class and 0 to majority class
    df['Class'] = index
    df['Class'] = df['Class'].replace({True:1,False:0})

    #split dataframe into Class and variables dataframes
    y = df['Class'].values.flatten()
    X = df.drop('Class', axis=1)
    X_new = []

    if cat_vars == False:
        smote = SMOTE(random_state=12)
        X_new, _ = smote.fit_resample(X,y)

    else:
        ind = [np.where(df.columns == x)[0][0] for x in cat_vars]

        smote_nc = SMOTENC(categorical_features=ind, random_state=12)
        X_new, _ = smote_nc.fit_sample(X,y)

    new_df = pd.DataFrame(X_new, columns=X.columns)

    # convert numerical columns to float
    if cat_vars:
        for c in new_df.copy():
            if c not in cat_vars:
                new_df[c]=new_df[c].astype(float)

    # return in long pandas dataframe format or one-hot encoded format
    if one_hot:
        return pd.get_dummies(new_df)
    else:
        return new_df

def fix_encoded_test(train_encoded, test_encoded):
    # fixs the test one-hot encoded dataset to have the same column variables as
    # the train one-hot encoded dataset
    # Input: train one-hot encoded and test one-hot encoded datasets.
    # Output: fixed test one-hot encoded dataset.

    import numpy as np

    not_in_test = []
    index = []
    for i,c in enumerate(train_encoded.columns):
        if c not in test_encoded:
            not_in_test.append(c)
            index.append(i)

    for i,c in enumerate(not_in_test):
        test_encoded.insert(loc=index[i],column=c,
                            value=np.zeros(len(test_encoded)))

    # reorder test sample columns to match order odf train sample
    test_encoded=test_encoded[train_encoded.columns]

    return test_encoded
