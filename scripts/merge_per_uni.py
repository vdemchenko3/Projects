# Python script for merging the cleaned performance and conversion csv files
#
# To execute the script (assuming you have Python 3.X installed) use
# python merge_per_uni.py uni_name
#
# The script assumes your cleaned data is in a folder called Clean_data/ one
# level above your script.
#
# The merged data will be written to the Clean_data/ folder

# Import required Python packages
import pandas as pd
import numpy as np
import function
import sys
import os.path

args = sys.argv
uni = str(args[1])

# Read in the cleaned performance and conversion files
perf = pd.read_csv("../Clean_data/Clean_performance_%s.csv" % uni)
conv = pd.read_csv("../Clean_data/Clean_conversion_%s.csv" % uni)

# Aggregate data in performance file. We do this to have unique IDs to merge
# with the conversion file. At this step we basically lose the country and city
# information as we have no way to combine this with the conversion data
perf_grouped = perf.groupby(['ID',
                           'Advertiser ID',
                           'Advertiser',
                           'Campaign ID',
                           'Campaign',
                           'Site ID (DCM)',
                           'Site (DCM)',
                           'Creative ID',
                           'Creative',
                           'Creative Type',
                           'Placement Pixel Size',
                           'Platform Type'])

# Having aggregated the data we need to perform an appropriate operation on the
# the columns containing numerical data. The operation performed is a summation.
# 
# In addition we also compute the number of days that each ID ran for so that
# we can create a new column called Duration which tells us the number of days
# per ID. We compute this separately for both the performance and conversion 
# datasets thus days_per_id_perf contains the number of days per ID for the 
# performance dataset
days_per_id_perf = perf_grouped['Date'].unique().reset_index()
days_per_id_perf = days_per_id_perf[['ID','Date']]

# Create a list containing the columns to aggregate over - basically all our 
# numerical data
cols_to_agg = ['Clicks','Impressions',
             'Active View: Viewable Impressions',
             'Active View: Measurable Impressions',
             'Active View: Eligible Impressions']

# Do the aggregation 
perf_grouped_df = perf_grouped[cols_to_agg].sum()
# Now reset the index on our aggregated performance
perf_grouped_df = perf_grouped_df.reset_index()

# Compute the Click Rate for our aggregated data
perf_grouped_df['Click Rate'] = perf_grouped_df['Clicks']/perf_grouped_df['Impressions']*100

# We need to correct our Click Rate values as follows:
# Clicks=0 and Impressions=0 produce np.nan. We replace them with 0.
# Clicks>0 and Impressions=0 produce np.inf. We replace them with np.nan
perf_grouped_df['Click Rate'].replace(to_replace=np.nan,value=0,inplace=True)
perf_grouped_df['Click Rate'].replace(to_replace=np.inf, value=np.nan,inplace=True)

# Now perform the aggregation for the conversion data set. Here we aggregate
# over the date as we have no way to link this to the performance data
conv_grouped = conv.groupby(['ID',
                         'Advertiser ID',
                         'Advertiser',
                         'Campaign ID',
                         'Campaign',
                         'Site ID (DCM)',
                         'Site (DCM)',
                         'Creative ID',
                         'Creative',
                         'Creative Type',
                         'Placement Pixel Size',
                         'Platform Type'])

# Compute the sum of the numeric columns (c.f. the performance data) and reset
# the index. Again we also compute the number of days per ID so that we can create
# our Duration column
days_per_id_conv = conv_grouped['Date'].unique().reset_index()
days_per_id_conv = days_per_id_conv[['ID','Date']]

# Create a list of the columns we need to aggregate over
cols_to_agg = ['Total Conversions',
             'Click-through Conversions',
             'View-through Conversions']

# Do the aggregation and reset the index
conv_grouped_df = conv_grouped[cols_to_agg].sum()
conv_grouped_df = conv_grouped_df.reset_index()

# Calculate the Duration column 
merged_days = days_per_id_conv.merge(days_per_id_perf, on='ID')
merged_days['Duration'] = merged_days['Date_x'].copy()
for index,row in merged_days.iterrows():
    a = row['Date_x']
    b = row['Date_y']
    merged_days.loc[index,'Duration'] = np.append(a,b)


# Having obtained the dates we basically work out the maximum duration 
# possible between the performance and conversion datasets
merged_days.drop(['Date_x','Date_y'],axis = 'columns', inplace=True)
merged_days['Duration'] = merged_days['Duration'].apply(np.unique)
merged_days['Duration'] = merged_days['Duration'].apply(pd.to_datetime)
### This makes sure that the dates are sorted so when you take the difference you're getting the first and last 'true' date
### If you get a dataset that's not sorted, you may get negative duration values
merged_days['Duration'] = merged_days['Duration'].apply(lambda x: (np.sort(x)[-1]-np.sort(x)[0]).astype('timedelta64[D]') / np.timedelta64(1, 'D') + 1)


# Join the performance and conversion datasets. We use the ID that we created
# and perform an inner join meaning that only indexes that are common to both
# the performance and conversion data sets will be included
merged_df = conv_grouped_df.merge(perf_grouped_df, on=['ID'], how='inner')

# Merge with the new Duration column 
merged_df = merged_df.merge(merged_days, on='ID')

# After the join we have a number of redundant columns as each cleaned data
# set contained these data. We create a list called cols_to_drop and use this
# to remove those columns from our data frame
cols_to_drop = ['Advertiser ID_y','Advertiser_y','Campaign ID_y',
                'Campaign_y','Site ID (DCM)_y','Site (DCM)_y',
                'Creative Type_y','Creative_y', 'Placement Pixel Size_y',
                'Platform Type_y','Creative ID_y']

# Drop the redundant columns
merged_df = merged_df.drop(cols_to_drop, axis='columns')

# Rename the _x columns removing the _x as having dropped the duplicate columns
# the _x in the column name is now meaningless.
merged_df.rename(columns={"Advertiser ID_x":"Advertiser ID",
                          "Advertiser_x":"Advertiser",
                          "Campaign ID_x":"Campaign ID",
                          "Campaign_x":"Campaign",
                          "Site ID (DCM)_x":"Site ID (DCM)",
                          "Site (DCM)_x":"Site (DCM)",
                          "Creative Type_x":"Creative Type",
                          "Placement Pixel Size_x":"Placement Pixel Size",
                          "Platform Type_x":"Platform Type",
                          "Creative ID_x":'Creative ID',
                          "Creative_x":"Creative"},
                 inplace=True)


# Creating 4 bins for Placement Pixel Size (Tracking, Banner, Leaderboards, MPUs):
merged_df['Placement Pixel Size Bin'] = merged_df['Placement Pixel Size'].copy()

placement_bins = {'1 x 1' : 'Tracking',
    '300 x 600' : 'Banner', '160 x 600' : 'Banner', '120 x 600' : 'Banner',
    '180 x 150' : 'Banner', '160 x 601' : 'Banner', '245 x 210' : 'Banner',
    '210 x 229' : 'Banner','320 x 480' : 'Banner','300 x 601' : 'Banner',
    '728 x 90' : 'Leaderboards', '320 x 50' : 'Leaderboards', '468 x 60' : 'Leaderboards',
    '970 x 250' : 'Leaderboards', '970 x 90' : 'Leaderboards', '320 x 100' : 'Leaderboards',
    '300 x 50' : 'Leaderboards', '650 x 100' : 'Leaderboards', '250 x 100' : 'Leaderboards',
    '600 x 250' : 'Leaderboards', '120 x 90' : 'Leaderboards', '580 x 400' : 'Leaderboards',
    '300 x 100' : 'Leaderboards', '600 x 90' : 'Leaderboards', '990 x 100' : 'Leaderboards',
    '992 x 115' : 'Leaderboards', '970 x 150' : 'Leaderboards','728 x 91' : 'Leaderboards',
    '994 x 118' : 'Leaderboards', '994 x 250' : 'Leaderboards','320 x 150' : 'Leaderboards',
    '940 x 90' : 'Leaderboards','1020 x 125' : 'Leaderboards','728 x 92' : 'Leaderboards',
    '425 x 325' : 'Leaderboards','970 x 93' : 'Leaderboards','360 x 300' : 'Leaderboards',
    '628 x 90' : 'Leaderboards','1055 x 90' : 'Leaderboards','980 x 90' : 'Leaderboards',
    '970 x 92' : 'Leaderboards','750 x 100' : 'Leaderboards',
    '300 x 250' : 'MPUs', '336 x 280' : 'MPUs', '250 x 250' : 'MPUs', '200 x 200' : 'MPUs',
    '300 x 251' : 'MPUs', '300 x 254' : 'MPUs', '640 x 640' : 'MPUs', '300 x 258' : 'MPUs',
    '320 x 250' : 'MPUs', '100 x 100' : 'MPUs', '300 x 253' : 'MPUs', '338 x 280' : 'MPUs',
    '300 x 252' : 'MPUs', '300 x 380' : 'MPUs','300 x 255' : 'MPUs','320 x 267' : 'MPUs',
    '300 x 300' : 'MPUs','328 x 250' : 'MPUs'}

merged_df['Placement Pixel Size Bin'].replace(placement_bins, inplace=True)

# dropping the 'Placement Pixel Size Bin' column from the merged_df:
merged_df.drop(['Placement Pixel Size'], axis=1)

# Round total conversions to nearest integer
merged_df['Total Conversions'] = round(merged_df['Total Conversions'])

print('Length of %s merged file is:' % uni, len(merged_df))

# Finally write out our merged data frame to a csv file
fn = "../Clean_data/Merged_data.csv"

# Checks if file exists and if not, creates it.  If it does, it appends to it
if os.path.isfile(fn):
  with open(fn, 'a') as f:
    merged_df.to_csv(f, index=False, header=False)
else:
  merged_df.to_csv(fn, index=False)
