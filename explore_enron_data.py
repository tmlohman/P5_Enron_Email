#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import sys
import os
import pickle
sys.path.append("tools/")
from helper_functions import get_outliers
import numpy as np
from collections import defaultdict



enron_data = pickle.load(open("final_project_dataset.pkl", "r"))


# print out possible features to explore
feature_list = []
for k in enron_data.values()[0]:
	feature_list.append(k)
	#print k
	
#outliers = get_outliers(feature_list, enron_data)
#count = 0
#for k,v in outliers.iteritems():
	#count += v
#print count


# get count of persons of interest and total persons
poi_list = []
for p in enron_data:
	if enron_data[p]['poi'] == True:
		poi_list.append(p)

print " "
print "Persons of interest: " + str(len(poi_list))

# how many email authors are there in the corpus?
print "total persons in corpus: " + str(len(enron_data)) + "\n"


# make dictionary for a given feature values and names
def make_dict(data, feature):
	new_dict = {}
	for p in data:
		if data[p][feature] != "NaN" and p != "TOTAL":
			new_dict[p] = data[p][feature]
	return new_dict

			
# get stats on a given feature
def get_stats(data):
	min_val = min(data)
	max_val = max(data)
	std = np.std(data)
	mean = np.mean(data)
	outliers = []
	for d in data:
		if d > (mean + 2*std) or d < (mean - 2*std):
			outliers.append(d)
	return min_val, max_val, std, mean, outliers

# get outliers for desired features	
def get_outliers(feature_list, data):
	outliers = defaultdict(int)
	for f in feature_list:
		if f != "poi":
			try:
				f_data = make_dict(data, f)
				print f
				print f + " outliers:"
				stats = get_stats(f_data.values())
				for outlier in stats[-1]:
					for name, value in f_data.iteritems():
						if value == outlier and name in poi_list:
							print name + ": "+ str(value) + " (poi)"
							outliers[name] += 1
						if value == outlier and name not in poi_list:
							print name + ": "+ str(value)
							outliers[name] += 1
				for name in enron_data:
					if name not in outliers.keys():
						outliers[name] = 0
				print " "
			except:
				print "feature " + f + " not numerical"  + "\n"


	return outliers
	
## get stats and outliers for desired features	
#def get_outliers(feature_list, enron_data):
	#outliers = defaultdict(int)
	#all_stats = {}
	#for f in feature_list:
		#if f != "poi":
			#try:
				#f_data = make_dict(enron_data, f)
				#print f
				#stats = get_stats(f_data.values())
				#all_stats[f] = stats
				#print f + " outliers:"
				#for outlier in stats[-1]:
					#for name, value in f_data.iteritems():
						#if value == outlier and name in poi_list:
							#print name + ": "+ str(value) + " (poi)"
							#outliers[name] += 1
						#if value == outlier:
							#print name + ": "+ str(value)
							#outliers[name] += 1
				#for name in enron_data:
					#if name not in outliers.keys():
						#outliers[name] = 0
				#print " "
			#except:
				#print "feature " + f + " not numerical"  + "\n"
	#return outliers, all_stats
		
outliers = get_outliers(feature_list, enron_data)	

	
for key, value in outliers.iteritems():
	if key in poi_list:
		print key + ": " + str(value) + " (poi)"
	else:
		print key + ": " + str(value)
		
count = 0
for value in outliers.values():
	count += value
print count	

#

