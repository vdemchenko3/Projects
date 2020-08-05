import sys
import os.path
from apply_model import *


def test_pix_bin(bin):
	### Function to make sure that the input for Ad Type is in the model

	if bin not in ad_type:
		print()
		print("One of the Advertisement Types that you've entered is not valid!")
		print(bin, "is not a valid Advertisement Type!")
		print("Only the following Advertisement Types are valid: ", ad_type)
		print()
		pix_bin = input('Please enter a valid Advertisement Type (no quotation marks): ')

		return pix_bin

def test_sites(site):
	### Function to make sure that the input for Site is in the model

	if site not in site_names:
		print()
		print("One of the sites that you've entered is not valid!")
		print(site, "is not a valid Site!")
		print("Only the following Sites are valid: ", site_names)
		print()
		print('Please enter "Rare_Site" (no quotations) if the Site you want is: ', rare_site_names)
		print()
		site = input('Please enter a valid Site (no quotation marks): ')

		return site


### Ad Types that the model was trained on
ad_type = ['MPUs', 'Leaderboards', 'Banner', 'Tracking']


### Sites that the model was trained on
site_names = ['The Complete University Guide', 'Google Display Network',
       'Rare_Site', 'UCAS', 'What Uni', 'Programmatic', 'Open Days',
       'YouTube', 'Hotcourses', 'Find a Masters', 'Prospects',
       'The Student Room', 'DAX', 'FoneMedia', 'Student Crowd',
       'LinkedIn', 'Postgraduate Search', 'Paid Search', 'Study Portals',
       'Facebook', 'Instagram', 'Twitter', 'All 4', 'THE Online']


### Sites that are in the Rare_Site category
rare_site_names = ['Education UK', 'Adtima', 'bamuk.com', 'a-n', 'The Guardian',
       'Spotify', 'Social Influencers', 'startups.co.uk',
       'Crack Magazine', 'Bristol 24/7', 'The Bristol Mag',
       'Creative Review', "What's On Bristol", 'South West Business UK',
       'Business Matters', 'Business Post',
       'GSP (Gmail Sponsored Promotions)', 'Gazette Live',
       'Educations.com', 'The Engineer', 'ingenium-ids.org',
       'Uni Taster Days', 'Snapchat', 'Tees Business', 'Reed',
       'Master Studies']



print('Hello SMRS guru!')
print('You can use this program to predict the Total Conversions using two models.')
print()
print('Model 1: uses Reach/day and Site Name')
print('Model 2: uses Reach/day, Site Name, and Advertisement Type (Leaderboard, Banner, etc.)')
print()

model = input("Please input 1 or 2 for which model you'd like and press the enter/return key: ")


### Reach and Site model
if model == '1':

	### Initiate model 1 with default parameter ie only Reach and Site
	M1 = Model()

	### Ask user for inputs
	reach_raw = input('Please input the Reach/day (e.g. 2 or 2,3 if multiple entries): ')
	reach = [float(x) for x in reach_raw.split(',')]

	site_raw = input('Please input the Site Name (e.g. What Uni or What Uni, Facebook if multiple entries) or press return/enter to see your options: ')
	site = [x.strip() for x in site_raw.split(',')]

	### Make sure user inputs are allowed
	for i in range(len(site)):
		while site[i] not in site_names:
			if site[i] in rare_site_names:
				site[i] = 'Rare_Site'
			else:
				site[i] = test_sites(site[i])

	print()
	print('The Reach chosen is: ', reach)
	print('The Site chosen is: ', site)

	### Use model to predict the Total Conversions / day using Reach/day and Site
	pred = M1.make_prediction(reach=reach, site=site, ad_type=None)

	print()
	print('The predicted Total Conversions/Day Model 1 is: ')
	print(pred)

### Reach, Site, and Ad Type model
elif model == '2':

	### Initiate model 2 with Ad Type added
	M2 = Model(use_ad_type=True)

	### Ask user for inputs
	reach_raw = input('Please input the Reach/day (e.g. 2 or 2,3 if multiple entries): ')
	reach = [float(x) for x in reach_raw.split(',')]

	site_raw = input('Please input the Site Name (e.g. What Uni or What Uni, Facebook if multiple entries) or press return/enter to see your options: ')
	site = [x.strip() for x in site_raw.split(',')]

	### Make sure user inputs are allowed
	for i in range(len(site)):
		while site[i] not in site_names:
			if site[i] in rare_site_names:
				site[i] = 'Rare_Site'
			else:
				site[i] = test_sites(site[i])

	pix_bin_raw = input('Please input the Advertisement Type (e.g. MPUs or MPUs, Banner if multiple entries) or press return/enter to see your options: ')
	pix_bin = [y.strip() for y in pix_bin_raw.split(',')]
	
	### Make sure user inputs are allowed
	for j in range(len(pix_bin)):
		if pix_bin[j] not in ad_type:
			pix_bin[j] = test_pix_bin(pix_bin[j])


	print()
	print('The Reach chosen is: ', reach)
	print('The Site chosen is: ', site)
	print('The Ad Type chosen is: ', pix_bin)


	### Use model to predict the Total Conversions / day using Reach/day, Site, and Ad Type
	pred = M2.make_prediction(reach=reach, site=site, ad_type=pix_bin)

	print()
	print('The predicted Total Conversions/Day for Model 2 is: ')
	print(pred)

else:
	sys.exit('No such model!  Please rerun program and input 1 or 2 when prompted for the model!')



