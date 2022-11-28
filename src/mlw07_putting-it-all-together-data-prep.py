# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 06:45:23 2022

Machine Learning Workshop Session 7: Putting It All Together: Data Prep

@author: Tim G.

"""

# Import packages
## You can add more packages as you need them.
import pandas as pd
import mlw


# Define constants and configuration information.
STUDY_NAME          = "bank-telemarketing"
DATA_FOLDER_NAME    = "../data/"
DATA_FILE_NAME      = "bank-telemarketing_train.csv"
FILE_SEPARATOR      = ";"

# YOUR TURN: Change the configuration information for your dataset.
## NOTE: Throughout this script, ALL_CAPS signifies configuration information.
## You probably will want to move *all* of the configuration info to the top
## of your script, eventually.


# Read the dataset.
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)

# YOUR TURN: Read your dataset and look at it in Spyder.
## Did you read the dataset successfully?  How do you know?
## What do you know about your dataset thus far?




# Are there any elements that you want to rename?
## RENAME_ELEMENTS = {'from':'to'}
RENAME_ELEMENTS = {'job':'occupation'}
if len(RENAME_ELEMENTS) > 0:
    working = working.rename(columns=RENAME_ELEMENTS)

# YOUR TURN: Rename any elements as you prefer.
## Are there any characters you can or should *NOT* use in element names?




# Are there any elements that you want to ignore?
IGNORE_ELEMENTS = ["duration"]
if len(IGNORE_ELEMENTS) > 0:
    working.drop(columns=IGNORE_ELEMENTS, inplace=True)

# YOUR TURN: Specify any elements you want to ignore.
## Note: this list may change during the course of your project.




# Is the target element binary?  Do you need to "map" it to 0/1?
TARGET_ELEMENT_NAME = 'y'
TARGET_MAPPING      = {'yes':1, 'no':0}
BINARY_TARGET_NAME  = 'subscribed'

working[BINARY_TARGET_NAME] = working[TARGET_ELEMENT_NAME].map(TARGET_MAPPING).astype('float')
working.drop(columns=TARGET_ELEMENT_NAME, inplace=True)

# YOUR TURN: Map the target element, if necessary.  
## What is the target rate?
## How many positive outcomes are present?



# Are there any rows where the target is missing?
original_row_count = len(working)
working = working.dropna(subset=[BINARY_TARGET_NAME])
revised_row_count  = len(working)

# YOUR TURN: Drop any rows where the target is missing.
## How many rows did you lose?
## Why drop these rows?  Is there any way to use them?




# Are there any columns that are useless?
## First, identify any columns consisting entirely of NaNs and drop them
all_nan = working.isna().all(axis=0)
to_drop = all_nan[all_nan].index.to_list()
if len(to_drop) > 0:
    print('\n\nThe following elements are entirely missing (NaN), and they will be dropped:', to_drop)
    working.drop(columns=to_drop, inplace=True)

# identify any columns that are constant (single-valued) and drop them
constant_elements = []
for element in working.columns.to_list():
    if not (working[element] != working[element].iloc[0]).any():
        constant_elements.append(element)
if len(constant_elements) > 0:
    print('\n\nThe following elements are constant, and they will be dropped:', constant_elements)
    working.drop(columns=constant_elements, inplace=True)

# YOUR TURN: Drop any elements that are useless.
## How many elements did you lose?  Which ones?
## Why drop these elements?  Is there any way to use them?




# Are there any numeric nominal elements?
## They look like numbers, but it doesn't make sense to apply math opeations to them.
NUMERIC_NOMINAL_ELEMENTS = ['previous']
MISSING_VALUE_CHARACTER  = '.'
for element in NUMERIC_NOMINAL_ELEMENTS:
    working[element] = working[element].astype(str)
    working[element].replace('nan', MISSING_VALUE_CHARACTER, inplace=True)

# YOUR TURN: Specify any numeric nominal elements.
## Why do you convert them from numbers to strings?  What happens if you don't?




# convert any object elements (string, nominal) to categorical, 
# add a '.' category for missing and replace any NaNs with '.'
success = mlw.convert__object_to_category(working)

# YOUR TURN: Convert any object elements to categorical.
## Why do you convert them?  What happens if you don't?



# Split the working dataset into a training subset and testing subset,
# each consisting of predictors (X) and a target (y)
TEST_SET_FRACTION = 0.5
RANDOM_SEED       = 62362
(predictors_train, predictors_test, target_train, target_test, success) = mlw.split__dataset(
                working, BINARY_TARGET_NAME,
                test_set_fraction = TEST_SET_FRACTION,
                random_seed       = RANDOM_SEED
                )

# YOUR TURN: Split your dataset into training and testing partitions.
## Why split the dataset?
## Why put 50% of the data into each partition?  Why not use a different fraction?
## How many positive outcomes are in each partition?  What is the target rate in each?
## Why use a random seed?




