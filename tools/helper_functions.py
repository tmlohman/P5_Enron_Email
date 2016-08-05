#!/usr/bin/python

""" 
    Helper functions 
    
"""

import numpy as np
from collections import defaultdict

# get list of features
def get_features(data):
	feature_list = []
	for k in data.values()[0]:
		feature_list.append(k)
	return feature_list

# make dictionary for a given feature values and names
def make_dict(data, feature):
	new_dict = {}
	for p in data:
		if data[p][feature] != "NaN" and p != "TOTAL":
			new_dict[p] = data[p][feature]
	return new_dict
		
# get stats on a given feature
def get_stats(data):
	import numpy as np
	std = np.std(data)
	mean = np.mean(data)
	outliers = []
	for d in data:
		if d > (mean + 2*std) or d < (mean - 2*std):
			outliers.append(d)
	return std, mean, outliers
	
# get outliers for desired features	
def get_outliers(feature_list, data):
	from collections import defaultdict
	outliers = defaultdict(int)
	all_stats = {}
	for f in feature_list:
		if f != "poi":
			try:
				f_data = make_dict(data, f)
				stats = get_stats(f_data.values())
				for outlier in stats[2]:
					for name, value in f_data.iteritems():
						if value == outlier:
							outliers[name] += 1
				for name in data:
					if name not in outliers.keys():
						outliers[name] = 0
			except:
				pass

	return outliers

		
	
