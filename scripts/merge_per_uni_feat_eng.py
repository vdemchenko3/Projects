import pandas as pd
import numpy as np
import function

all_uni_df = pd.read_csv("../Clean_data/Merged_data.csv")

# FEATURE ENGINEERING
# Create the Reach variable. We define this variable as the Impressions for
# creatives can have Impressions and as Clicks for the ones that, by
# definition, cannot have impressions.


df_with_reach = function.calculate_reach(all_uni_df)
all_uni_df['Reach'] = df_with_reach['Reach']

# Adding the 'Site Bin' column. It only bins the sites with count less than
# a certain number, the rest of the sites remain as they are.
values = all_uni_df['Site (DCM)'].value_counts().reset_index().values
site_count_df = pd.DataFrame(values, columns=['site', 'count'])

### This number is arbitrarily hardcoded (can change)
site_count_threshold = 8
site_condition = site_count_df['count'] < site_count_threshold

rare_sites = site_count_df.loc[site_condition , 'site'].tolist()
rare_sites_dict = { i : 'Rare_Site' for i in rare_sites }

all_uni_df['Site Bin'] = all_uni_df['Site (DCM)']
all_uni_df['Site Bin'].replace(rare_sites_dict, inplace=True)

#remove Natural Search
cond1 = all_uni_df['Site Bin'] != 'Natural Search'
all_uni_df = all_uni_df.loc[cond1]

all_uni_df.to_csv("../Clean_data/Merged_data.csv", index=False)