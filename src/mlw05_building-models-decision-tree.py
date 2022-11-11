# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 07:34:24 2022

Machine Learning Workshop Session 5: Building Models: Decision Tree

@author: Tim G.

"""

# import packages
import pandas            as pd
import matplotlib        as plt
import mlw                                          # this is our own "war chest"
from sklearn.tree    import DecisionTreeClassifier  # for building the classifier
from sklearn         import metrics                 # for the accuracy calculation
from sklearn         import tree                    # for the tree visualization



# define constants and configuration information
STUDY_NAME          = "bank-telemarketing"
DATA_FOLDER_NAME    = "../data/"
DATA_FILE_NAME      = "bank-telemarketing_train.csv"
FILE_SEPARATOR      = ";"

TARGET_ELEMENT_NAME = 'y'
TARGET_MAPPING      = {'yes':1, 'no':0}
BINARY_TARGET_NAME  = 'subscribed'

# see https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
TREE_CRITERION         = "entropy"                  # this block contains decision
TREE_MAX_DEPTH         = 4                          # tree parameters
TREE_MIN_SAMPLES_SPLIT = 500
TREE_MIN_SAMPLES_LEAF  = 250




# Open the bank-telemarketing dataset...
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)



# Prep for building a decision tree model:
## 1. We can only use numeric elements as predictors. 
candidates = working.select_dtypes(include='number').columns.tolist()

## 2. The target must be binary.
working[BINARY_TARGET_NAME] = working[TARGET_ELEMENT_NAME].map(TARGET_MAPPING).astype('float')

## 3. The sklearn package *requires* that we separate the predictors from the target.
predictors = working[candidates]
target     = working[BINARY_TARGET_NAME]


# Fit the decision tree model to the data using the DecisionTreeClassifier method
## First, create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion        =TREE_CRITERION, 
                             max_depth        =TREE_MAX_DEPTH, 
                             min_samples_split=TREE_MIN_SAMPLES_SPLIT, 
                             min_samples_leaf =TREE_MIN_SAMPLES_LEAF)

## Next, fit (train) the classifier to the data
clf = clf.fit(predictors,target)

# Let's look at the tree!!!
tree.plot_tree(clf);







## We need to *drastically* improve the tree diagram to make it usable!
### Let's start by creating a "parameter" block
### see https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree
### for details on the available parameters
TREE_PLOT_WIDTH      = 32
TREE_PLOT_HEIGHT     = 16
TREE_PLOT_LABEL      = "root"
TREE_PLOT_FILLED     = True
TREE_PLOT_ROUNDED    = True
TREE_PLOT_PROPORTION = True

## Let's try again
plt.rcParams["figure.figsize"] = (TREE_PLOT_WIDTH, TREE_PLOT_HEIGHT)
tree.plot_tree(clf, feature_names=candidates, 
               label  =TREE_PLOT_LABEL, 
               filled =TREE_PLOT_FILLED, 
               rounded=TREE_PLOT_ROUNDED,
               proportion=True);

### YOUR TURN: What do you see?  How do you *read* the tree?


## One more time w/ a useful variation
tree.plot_tree(clf, feature_names=candidates, 
               label  =TREE_PLOT_LABEL, 
               filled =TREE_PLOT_FILLED, 
               rounded=TREE_PLOT_ROUNDED,
               proportion=False);

### YOUR TURN: Review the decision tree results: what do you notice?
#### Which predictors are "most important"?

# Apply the model, that is, *score* the data using the model.
target_class = clf.predict(predictors)        # this is a 0/1 "hard" classification
target_class = pd.Series(target_class)
target_class.name = "target_class"


target_prob  = clf.predict_proba(predictors)  # this is the target probability (fraction)
target_prob  = pd.Series(target_prob[:,1])
target_prob.name = "target_prob"

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(target, target_class))  # this is one metric

## YOUR TURN: Do you think this metric is useful?  Why/not?
## How well would you expect to do on a problem like this?

## What else could you do?
print(pd.crosstab(target, target_class))                         # the "confusion matrix"


# Plot the results as an ROC curve
## NOTE: We're using a function from our very own mlw.py!!
tree_auc = mlw.plot__roc_curve(target, target_prob)

## YOUR TURN: How do the results compare to your expectations?
### Why do you think they are better/worse?
### How might you want to change the model to improve it?

# YOUR TURN: Remove one or more predictors and re-run the model.
## How does the performance compare?
## Change one tree parameter, say, TREE_MAX_DEPTH.
## What happens?

# YOUR TURN: 'Smart" binning.  Keep only a single predictor and re-run the model.
## How well does the predictor single do by itself?
## Repeat for several different predictors.  Which ones are the best?

# CHALLENGE: write a function to produce a *usable* tree diagram.




predictors.drop(columns=['duration'], inplace=True)   # to drop a column(s) from a dataframe
candidates.remove('duration')                         # to remove an item(s) from a list



