#!/usr/bin/python

import sys
import os
import pickle
sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from helper_functions import get_features
from helper_functions import get_outliers


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# selecting all features, which I will later prune using feature selection
features_list = (['poi','salary', 'to_messages', 'deferral_payments', 
					'total_payments', 'exercised_stock_options', 'bonus', 
					'restricted_stock', 'shared_receipt_with_poi', 
					'restricted_stock_deferred', 'total_stock_value',
					'expenses', 'loan_advances', 'from_messages', 'other', 
					'from_this_person_to_poi', 'director_fees', 
					'deferred_income', 'long_term_incentive', 
					'from_poi_to_this_person', 'outlier_count']) # You will need to use more features

#features_list = (['poi','bonus', 'salary'])


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Removing TOTAL values. Other outliers may be indication of POI and 
# therefore should not be removed.
del data_dict['TOTAL']


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# get outlier counts from helper functions
all_feature_list = get_features(data_dict)
outliers = get_outliers(all_feature_list, data_dict)

my_dataset = data_dict

# make a list of all persons in the dataset, add their outlier counts
persons_list = []
for p in my_dataset:
	persons_list.append(p)
	
for person in persons_list:
	my_dataset[person]["outlier_count"] = outliers[person]

all_feature_list.append('outlier_count')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
def decision_tree_clf(features_train, labels_train):
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	return clf.fit(features_train, labels_train)

def naive_bayes_clf(features_train, labels_train):
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	return clf.fit(features_train, labels_train)

def svm_clf(features_train, labels_train):
	from sklearn import svm
	clf = svm.SVC()
	return clf.fit(features_train, labels_train)

clf = svm_clf(features_train, labels_train)
pred = clf.predict(features_test)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


# run SelectKBest on all features to get some sense of how to proceed
#from sklearn.feature_selection import SelectKBest
#selected = SelectKBest(k = 'all')
#selected.fit(features_train, labels_train)

#scores = selected.scores_
#feature_scores = {}
#for s in range(len(features_list)-1):
	#feature_scores[features_list[s + 1]] = scores[s]
	
#for f in sorted(feature_scores, key=feature_scores.get, reverse=True):
  #print f, feature_scores[f]

from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score

clf = make_pipeline(SelectKBest(), GaussianNB())
param_grid = dict(selectkbest__k = [1,2,3,4,5,6,7,8,9,10,11])
grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=10, scoring = 'precision')
grid_search.fit(features_train, labels_train)
clf = (grid_search.best_estimator_)






### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
