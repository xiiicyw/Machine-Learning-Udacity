#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Print out the number of data points (people).
print len(enron_data)

# Print out the number of available features of each person.
print len(enron_data[enron_data.keys()[0]])

# Print out the number of POIs existing in the E+F dataset.
POI_count = 0
for element in enron_data:
    if enron_data[element]["poi"] == 1:
        POI_count += 1
print POI_count

# Check all the individual people names.
print enron_data.keys()

# Check all the feature names.
print enron_data[enron_data.keys()[0]]

# The feature name of total value of stock is "total_stock_value".
# Print out the total value of stock belonging to James Prentice.
print enron_data["PRENTICE JAMES"]["total_stock_value"]

# The feature name of number of email messages sent to POIs is "from_this_person_to_poi".
# Print out the number of email messages sent from Wesley Colwell to persons of interest.
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# The feature name of the value of stock options is "exercised_stock_options".
# Print out the value of stock options exercised by Jeffrey Skilling.
# The full name listed is "SKILLING JEFFREY K".
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

# Find out who in Lay, Skilling and Fastow took home the most money, and the value of total_payments.
print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]

# Find out how many people have a quantified salary. And a known email address.
# Feature names: "salary", "email_address".
salary_count = 0
for element in enron_data:
    if enron_data[element]["salary"] != "NaN":
        salary_count += 1
print salary_count

email_address_count = 0
for element in enron_data:
    if enron_data[element]["email_address"] != "NaN":
        email_address_count += 1
print email_address_count

# Find out how many people in the E+F dataset have "NaN" for their total payments.
# What percentage of people in the dataset as a whole is this.
total_payments_count = 0
for element in enron_data:
    if enron_data[element]["total_payments"] == "NaN":
        total_payments_count += 1
print total_payments_count
print float(total_payments_count) / len(enron_data)

# How many POIs in the E+F dataset have "NaN" for their total payments?
# What percentage of POI's as a whole is this?
POI_total_payments_count = 0
for element in enron_data:
    if enron_data[element]["poi"] == 1:
        if enron_data[element]["total_payments"] == "NaN":
            POI_total_payments_count += 1
print POI_total_payments_count
print float(POI_total_payments_count) / len(enron_data)

# What's the dictionary key of the biggest Enron outlier.
# This question appeared in Lesson 7.
for element in enron_data:
    if enron_data[element]["salary"] >= 20000000 and enron_data[element]["salary"] != "NaN":
        print element

# What are the names associated with new outliers?
# This question appeared in Lesson 7.
for element in enron_data:
    if enron_data[element]["bonus"] >= 5000000 and enron_data[element]["bonus"] != "NaN"\
            and enron_data[element]["salary"] >= 1000000 and enron_data[element]["salary"] != "NaN"\
            and element != "TOTAL":
        print element