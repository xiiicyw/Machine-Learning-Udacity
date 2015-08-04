#!/usr/bin/python

import sys
import pickle
import panda as pd

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary']
# You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Generate all feature names for better understanding of featuring features.
for element in data_dict[data_dict.keys()[0]].keys():
    print element

### Generate all elements of data_dict for better understanding of possible outliers.
for element in data_dict:
    print element

### Classify all features into categories.
poi = ["poi"]

### Email_address is deleted from features. It is no use for classification here.
### "to_messages", "from_messages", "long_term_incentive", "restricted_stock", "restricted_stock_deferred" are removed.
features_email = ["shared_receipt_with_poi",
                  "from_this_person_to_poi",
                  "from_poi_to_this_person"]

features_financial = ["salary",
                      "deferral_payments",
                      "total_payments",
                      "exercised_stock_options",
                      "bonus",
                      "total_stock_value",
                      "expenses",
                      "loan_advances",
                      "other",
                      "director_fees",
                      "deferred_income"]

### Task 2: Remove outliers
### Manually check the pdf file and dictionary elements output to remove the outliers.
### LOCKHART EUGENE E has NaNs for all features.
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for i in outliers:
    data_dict.pop(i, 0)

### Task 3: Create new feature(s)
### Create new features based on the ratio of poi related messages and all messages.
for element in data_dict:
    from_poi_to_this_person = data_dict[element]['from_poi_to_this_person']
    from_this_person_to_poi = data_dict[element]['from_this_person_to_poi']
    shared_receipt_with_poi = data_dict[element]["shared_receipt_with_poi"]

    to_messages = data_dict[element]['to_messages']
    from_messages = data_dict[element]['from_messages']

    if from_this_person_to_poi != "NaN" and from_messages != "NaN":
        to_poi_ratio = float(from_this_person_to_poi) / from_messages
    else:
        to_poi_ratio = 0.

    if from_poi_to_this_person != "NaN" and to_messages != "NaN":
        from_poi_ratio = float(from_poi_to_this_person) / to_messages
    else:
        from_poi_ratio = 0.

    if from_this_person_to_poi != "NaN" and from_messages != "NaN" and \
                    from_poi_to_this_person != "NaN" and to_messages != "NaN":
        total_messages = to_messages + from_messages
        poi_related_messages = shared_receipt_with_poi + from_poi_to_this_person + from_this_person_to_poi
        poi_related_ratio = float(poi_related_messages) / total_messages
    else:
        poi_related_ratio = 0.

    data_dict[element]['to_poi_ratio'] = to_poi_ratio
    data_dict[element]['from_poi_ratio'] = from_poi_ratio
    data_dict[element]['poi_related_ratio'] = poi_related_ratio

features_new = ["to_poi_ratio", "from_poi_ratio", "poi_related_ratio"]

### Build the full feature list.
features_list = poi + features_financial + features_email + features_new

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Split the dataset.
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers

if __name__ == '__main__':
    pca = PCA()
    scaler = MinMaxScaler()
    KBest = SelectKBest(score_func=f_classif)

    # Uncomment the following lines to run linear support vector machine classifier.
    '''
    # Linear SVM
    clf = LinearSVC()
    clf_pipeline = Pipeline([
        ('KBest', KBest),
        ('PCA', pca),
        ('MinMaxScaler', scaler),
        ('Classification', clf)
    ])
    parameters = {
        'KBest__k': (5, 10, 15, 17, 'all'),
        'PCA__n_components': (2, 3, 4),
        'PCA__whiten': (True, False),
        'Classification__C': (0.00001, 0.0001, 0.01, 0.1, 1, 100, 10000),
        'Classification__class_weight': ('auto',)
    }
    '''

    # Uncomment the following lines to run linear decision tree classifier.
    '''
    # Decision Trees
    clf = DecisionTreeClassifier()
    clf_pipeline = Pipeline([
        ('KBest', KBest),
        ('PCA', pca),
        ('Classification', clf)
    ])
    parameters = {
        'KBest__k': (6, 10, 15, 17, 'all'),
        'PCA__n_components': (2, 4, 6),
        'PCA__whiten': (False, True),
        'Classification__min_samples_split': (2, 20, 40),
        'Classification__criterion': ('entropy', 'gini'),
        'Classification__random_state': (42,)
    }
    '''

    # Uncomment the following lines to run linear Gaussian Naive Bayes classifier.
    '''
    # Naive Bayes
    clf = GaussianNB()
    clf_pipeline = Pipeline([
        ('KBest', KBest),
        ('PCA', pca),
        ('Classification', clf)
    ])
    parameters = {
        'KBest__k': (10, 13, 15, 17, 'all'),
        'PCA__n_components': (2, 3, 4, 5, 10),
        'PCA__whiten': (False, True),
    }
    '''

    # Uncomment the following lines to run logistic regression classifier.
    '''
    # Logistic Regression
    clf = LogisticRegression()
    clf_pipeline = Pipeline([
        ('KBest', KBest),
        ('PCA', pca),
        ('Classification', clf)
    ])
    parameters = {
        'KBest__k': (10, 13, 15, 17, 'all'),
        'PCA__n_components': (3, 5, 6, 7, 8, 9, 10),
        'PCA__whiten': (True, False),
        'Classification__C': (0.01, 0.1, 1, 10, 1000, 10000),
        'Classification__tol': (10 ** -1, 10 ** -5, 10 ** -10),
        'Classification__class_weight': ('auto',)
    }
    '''

    # The best result should be able to achieved by logistic regression.
    # Accuracy: 0.79307	Precision: 0.31746	Recall: 0.48000	F1: 0.38217	F2: 0.43541

    grid_search = GridSearchCV(clf_pipeline, parameters, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(features_train, labels_train)
    clf = grid_search.best_estimator_

    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    test_classifier(clf, my_dataset, features_list)

    ### Dump your classifier, dataset, and features_list so
    ### anyone can run/check your results.

    ### dump_classifier_and_data(clf, my_dataset, features_list)
