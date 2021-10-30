
# coding: utf-8

# In[7]:

import math

import random

import csv

from __future__ import division

# Reading the CSV file using the csv reader object 

# I have worked with 3 data sets which are as follows, please remove the comment from the dataset to be run and add comment to the ones to be ignored

# By default i have set immunotherapy dataset to be iterated upon

#with open('C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork 1\Decision Tree\data_banknote_authentication.csv', "rb") as fi:

#with open('C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork 1\Decision Tree\cryotherapy.csv', "rb") as fi:

with open('C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork 1\Decision Tree\immunotherapy.csv', "rb") as fi:
    reader = csv.reader(fi)

# Stores the field/column names    
    header = reader.next()
	
# Using the CSV.QUOTE_NONNUMERIC function to convert strings to numerical type    
    reader_n = csv.reader(fi,quoting=csv.QUOTE_NONNUMERIC) 
	
# Extracts the dataset    
    data = [row for row in reader_n]

my_randoms = random.sample(xrange(1,len(data)), int(len(data)/10))

test_data=[]
train_data=[]

for i in my_randoms:
    test_data.append(data[i])
	
for i in range(0,len(data)):
    #for j in range(0,len(my_randoms)): 
        if i not in (my_randoms):          
            train_data.append(data[i])
			
			
		   
# The dataset is partitioned based on condition class objects 
class Condition:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this condition.
        val = example[self.column]
        if isinstance(val, int) or isinstance(val, float):
            return val >= self.value

    def __repr__(self):
        # This is just a helper method to print
        # the condition in a readable format.
        if isinstance(self.value, int) or isinstance(self.value, float):
            expression = ">="
        return "Is %s %s %s?" % (
            header[self.column], expression, str(self.value))

#This module partitions the data set to the true branch and false branch based on some conditions generated
def partition(rows, condition):

    t_list, f_list = [], []
    for row in rows:
        if condition.match(row):
            t_list.append(row)
        else:
            f_list.append(row)
    return t_list, f_list

# Evaluates number of each type of labels in a dataset
def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # Because the class label is on the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

#This module calculates the GINI Impurity

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

#This module calculates the information gain

def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / int((len(left) + len(right)))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

#This module calculates the Entropy
	
def entropy(rows):
    counts = class_counts(rows)
    impurity = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity += prob_of_lbl * math.log(prob_of_lbl,2)
    return -1 * impurity

#This module calculates the information gain

def info_gain_entropy(left, right, current_uncertainty):
    p = float(len(left)) / int((len(left) + len(right)))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


#This module is used to determine the best condition with respect to every feature using gain, we can choose between Gini or Entropy based on the argument provided in function call

def find_best_split(rows,method="gini"):

    best_gain = 0  
    best_condition = None  
    if method=="gini":
        current_uncertainty = gini(rows)
    else:
        current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            condition = Condition(col, val)

            
            t_list, f_list = partition(rows, condition)


            if len(t_list) == 0 or len(f_list) == 0:
                continue

            # Information gain is calculated
            if method == "gini":
                gain = info_gain(t_list, f_list, current_uncertainty)
            else:
                gain = info_gain_entropy(t_list, f_list, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_condition = gain, condition

    return best_gain, best_condition

# Defining a class for each node type
	
# class defining a leaf node in the tree
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

# class defining a decision node in the tree
class Decision_Node:

    def __init__(self,
                 condition,
                 true_branch,
                 false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

# This module partitions the dataset and returns the condition with maximum information gain.

# The attribute value method is used to specify whether to use GINI or Entropy
		
def build_tree(rows,method):

    gain, condition = find_best_split(rows,method)

    # if gain is 0, the leaf is returned
    if gain == 0:
        return Leaf(rows)

    # Partitioning
    t_list, f_list = partition(rows, condition)

    # Recursive function calls
    true_branch = build_tree(t_list,method)

    false_branch = build_tree(f_list,method)

    # Return a Condition node.

    return Decision_Node(condition, true_branch, false_branch)

#Module to perform classification
def classify(row, node):

    # condition to check if node is a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Conditional statements on whether to follow the true branch or the false branch.
    if node.condition.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

# Module to print the leaf
def print_lastnode(counts):
    
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(int(counts[lbl] / total * 100))) + "%"
    return probs

# Main function	
if __name__ == '__main__':

    my_tree = build_tree(train_data,"entropy") # Can input gini or entropy as argument to the build tree function to specify the methodology to be used

# Calculating the accuracy of the model built
	
p=[]
predicted=[]
actual=[]    
for row in test_data:
    actual.append(row[-1])  
    p.append(print_lastnode(classify(row, my_tree)))
    print("Actual: %s. Predicted: %s" % (row[-1], print_lastnode(classify(row, my_tree))))
     
for keys in range(0,len(p)):
    for key, value in (p[keys]).iteritems() :
        predicted.append(key)

# Correctly predicted values		
correct=[i for i, j in zip(actual, predicted) if i == j]
# InCorrectly predicted values
incorrect=[i for i, j in zip(actual, predicted) if i != j]
accuracy= (len(correct)/(len(correct)+len(incorrect)))*100

print(" The Accuracy of the model in this iteration is : %s%% " %(accuracy))

		


# In[ ]:



