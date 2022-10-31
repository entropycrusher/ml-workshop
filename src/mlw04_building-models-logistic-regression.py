# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 07:14:23 2022

Machine Learning Workshop Session 4

@author: Tim G.

"""

# import packages
import pandas            as pd
import statsmodels.api   as sm           # this package includes logistic regression
import mlw                               # see mlw.py.  this is our own "war chest"

# define constants and configuration information
STUDY_NAME          = "bank-telemarketing"
DATA_FOLDER_NAME    = "../data/"
DATA_FILE_NAME      = "bank-telemarketing_train.csv"
FILE_SEPARATOR      = ";"
TARGET_ELEMENT_NAME = 'y'
TARGET_MAPPING      = {'yes':1, 'no':0}
BINARY_TARGET_NAME  = 'subscribed'
INTERCEPT_NAME      = 'intercept'        # name for the constant in the logistic regression
INTERCEPT_VALUE     = 1.0                # value for the constant




# Open the bank-telemarketing dataset...
## Note: the bank-telemarketing has semi-colons as separators, despite the .csv extension
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)


# Prep for building a logistic regression model:
## 1. We can only use numeric elements as predictors. 
## Let's create the list of potential candidates.
## (In a future session, we'll learn how to include categorical predictors.)
candidates = working.select_dtypes(include='number').columns.tolist()

## 2. The target must be binary.
## As we did previously, we can use the 'map' method to create a binary target.
working[BINARY_TARGET_NAME] = working[TARGET_ELEMENT_NAME].map(TARGET_MAPPING).astype('float')

## 3. The logistic package *requires* that we separate the predictors from the target.
## 3.1 Create a Dataframe that contains *only* the predictors.
predictors = working[candidates]

## 3.2 Create a Series (a dataframe with only one column) with *only* the target
target = working[BINARY_TARGET_NAME]

## 4. Add an *intercept* column (a constant) to the predictors and to the list of candidates
### YOUR TURN: Why would you want an input that is constant?
intercept  = pd.Series(INTERCEPT_VALUE, index=range(len(predictors)), name=INTERCEPT_NAME)
predictors[INTERCEPT_NAME] = INTERCEPT_VALUE
#predictors = pd.concat([predictors, intercept], axis=1)
candidates.append('intercept')

# Fit the logistic regression model to the data
## Use the sm (statsmodels.api) package
logit_model=sm.Logit(target,predictors[candidates])   # first, create the logistic "object"
logistic_regression=logit_model.fit(disp=False)       # then, you can build/fit the model

## Did anything happen?  Here's how you can display the results
### Quick tip: Use the dir() function to see all the attributes and methods for an object
print('\nDiagnostics')
print(logistic_regression.summary())
print('\nElement\t\tp-value')
print(logistic_regression.pvalues)

### YOUR TURN: Review the logistic results: what do you notice?
#### Which predictors are "most important"?
#### What does p-value (or P>|z|) mean?  What is a "good" p-value?

# Apply the model, that is, *score* the data using the model.
estimates = logistic_regression.predict(predictors)

## YOUR TURN: How do we know if the predictions are any good?
### And how well would you expect to do on a problem like this?
### Note: pseudo-r-squared

# Plot the results as an ROC curve
## NOTE: We're using a function from our very own mlw.py!!
logistic_auc = mlw.plot__roc_curve(target, estimates)

## YOUR TURN: What do the axes of the ROC curve mean?
### How do the results compare to your expectations?
### Why do you think they are better/worse?

# YOUR TURN: Remove one or more predictors and re-run the model.
## How does the performance compare?
## How will you keep track of your models, results, and plots as you make changes?

## HELP: you might find the following lines helpful
predictors.drop(columns=['blah'], inplace=True)   # to drop a column(s) from a dataframe
candidates.remove('blah')                         # to remove an item(s) from a list

## HELP: demo using Ctl-Alt-r to remove all variables and Ctl-l to clear the console.


## YOUR TURN: Should you make all of the drops at once, or one-at-a-time?
## What else might you do to improve model performance?



# CHALLENGE: write a function to remove predictors one-at-a-time
## until only those that are significant remain in the model.




