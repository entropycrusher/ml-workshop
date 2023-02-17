# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:46:23 2023

Machine Learning Workshop: machine learning workshop 10: building models

@author: Tim G.

"""

# Import packages
## You can add more packages as you need them.
import pandas            as pd
import matplotlib.pyplot as plt
import mlw
import pickle

# Define constants and configuration information.
STUDY_NAME          = "bank-telemarketing"
DATA_FOLDER_NAME    = "../data/"
FIGS_FOLDER         = "../figs/"
BENCHMARK_FOLDER    = "../bmrk/"

DATA_FILE_NAME      = "bank-telemarketing_train.csv"
FILE_SEPARATOR      = ";"
MISSING_VALUE_CHARACTER  = '.'

QUANTIZED_SUFFIX = "_q"
RATE_SUFFIX      = "_r"
P_VALUE          = 0.0001     # Note that we are NOT using .05 as our p-value.

# Import the oracle from the 'pickle' file
oracle_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_oracle.pkl")
fo = open(oracle_filename, 'rb')
oracle = pickle.load(fo)
fo.close()


# Read the dataset.
if len(oracle['config']['element_names']) > 0:
    working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR,
                        header=None, names=oracle['config']['element_names']
                        )
else:
    working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)


# Rename elements if needed
if len(oracle['config']['rename_elements']) > 0:
    working = working.rename(columns=oracle['config']['rename_elements'])

# Ignore elements if needed
if len(oracle['config']['ignore_elements']) > 0:
    working.drop(columns=oracle['config']['ignore_elements'], inplace=True)


# Handle numeric nominal elements, if needed
for element in oracle['config']['numeric_nominal_elements']:
    working[element] = working[element].astype(str)
    working[element].replace('nan', MISSING_VALUE_CHARACTER, inplace=True)

# Convert any object elements (string, nominal) to categorical, 
# add a '.' category for missing and replace any NaNs with '.'
success = mlw.convert__object_to_category(working)

# If any, cycle thru the bin-able elements and apply the bin boundaries
working = mlw.apply__bin_boundaries_to_all_numeric(working, 
                                                   oracle['bin_boundaries'],
                                                   suffix=QUANTIZED_SUFFIX
                                                   )

# If any, cycle thru the rate-able elements and apply the rate tables
working = mlw.apply__rate_tables_to_all_category(working, 
                                                 oracle['rate_tables'], 
                                                 oracle['target_rate'],
                                                 suffix=RATE_SUFFIX
                                                 )


# Produce the predictions for the training data using the fitted model
ESTIMATE_NAME = 'score'
ROW_ID        = 'row_id'
(working, estimate_train, success) = mlw.apply__logistic_regression(
                                                working,
                                                oracle['logistic_regression_model']
                                                )
estimate_train.name = ESTIMATE_NAME  # give the estimates Series a name


# Plot the distribution of scores
estimate_train.plot.hist(bins=25)
title_string = 'What does the distribution of scores look like?'
plt.title(title_string)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()


# Export the scored datafile
scored_data_filename = BENCHMARK_FOLDER + STUDY_NAME + "_scores.tab"
success = mlw.export__dataframe(pd.DataFrame(zip(working.index, estimate_train),
                                             columns=[ROW_ID, ESTIMATE_NAME]), 
                                             scored_data_filename)


