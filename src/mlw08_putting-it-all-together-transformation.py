# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 06:41:23 2022

Machine Learning Workshop Session 8: Putting It All Together: Transformation

@author: Tim G.

"""

# Import packages
## You can add more packages as you need them.
import mlw


# Define constants and configuration information.
STUDY_NAME       = "bank-telemarketing"
DATA_FOLDER_NAME = "../data/"
DATA_FILE_NAME   = "bank-telemarketing_train.csv"
FILE_SEPARATOR   = ";"

IGNORE_ELEMENTS          = ["duration"]
TARGET_ELEMENT_NAME      = 'y'
TARGET_MAPPING           = {'yes':1, 'no':0}
BINARY_TARGET_NAME       = 'subscribed'

# Run standard data prep (per workshop #7)
(working, target_name, 
 predictors_train, predictors_test, 
 target_train, target_test) = mlw.run__standard_prep(
                        data_file_name          =DATA_FILE_NAME,
                        file_separator          =FILE_SEPARATOR,
                        study_name              =STUDY_NAME,
                        ignore_elements         =IGNORE_ELEMENTS,
                        target_element_name     =TARGET_ELEMENT_NAME,
                        target_mapping          =TARGET_MAPPING,
                        binary_target_name      =BINARY_TARGET_NAME
                        )
 

     
     
# If any numeric elements exist, compute "smart" bin boundaries for them.
TREE_BIN_CRITERION = 'entropy'
TREE_BIN_DEPTH     = 2
TREE_BIN_MIN_LEAF  = 1000
TREE_PLOT          = True

bin_boundaries = mlw.compute__bin_boundaries_for_all_numeric(predictors_train, target_train,
                                           criterion=TREE_BIN_CRITERION,
                                           depth    =TREE_BIN_DEPTH,
                                           min_leaf =TREE_BIN_MIN_LEAF,
                                           tree_plot=TREE_PLOT
                                           )
# YOUR TURN: Compute the bin boundaries for your numeric elements.
## Look at your trees and the dictionary of bin boundaries.
## Pick out an element of interest and tell us what you see...
## Are any of the bin boundaries [-inf, inf]?  What does that mean?


# Cycle thru the bin-able elements and apply the bin boundaries
QUANTIZED_SUFFIX = "_q"
predictors_train = mlw.apply__bin_boundaries_to_all_numeric(predictors_train, bin_boundaries,
                                           suffix=QUANTIZED_SUFFIX
                                           )
# YOUR TURN: Apply the bin boundaries to your numeric elements.
## Look at your predictors.  
## How many new predictor columns did you create?  
## How can you identify them?
## What is the data type for the new predictors?
## How often does each interval/category appear?
## Why two steps, one to compute bin boundaries and another to apply them?
## Why not just combine them into a single step?


# Compute the target rate
target_rate = mlw.compute__target_rate(target_train)

# Estimate rates for all of the category elements
P_VALUE = 0.0001     # Note that we are NOT using .05 as our p-value.
rate_tables = mlw.compute__rate_tables_for_all_category(predictors_train, target_train, target_rate,
                                           target_p_value           =P_VALUE,
                                           confidence_limits_p_value=P_VALUE
                                           )
# YOUR TURN: Compute the rate tables for your category elements.
## Look at a rate table for a binned element and compare it to the decision tree.
## Which categories are predictive/not?
## Pick out an element of interest and tell us what you see...
## Are any of your elements useless?




# if any, cycle thru the rate-able elements and apply the rate tables
RATE_SUFFIX = "_r"
predictors_train = mlw.apply__rate_tables_to_all_category(predictors_train, rate_tables, target_rate,
                                    suffix=RATE_SUFFIX
                                    )
# YOUR TURN: Apply the rate tables to your category elements.
## Look at your predictors.  
## How many new predictor columns did you create?
## How can you identify them?  Which ones were numeric to start with?
## What is the data type for the new predictors?

# CHALLENGE:
## Why did we change numeric elements to categories and back to numbers again?
## What about missing values?  What happened to them?

########## STOPPED HERE ########################################################



# Save the bin boundaries as a benchmark file
BENCHMARK_FOLDER = "../bmrk/"
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_bin-boundaries.tab")
success = mlw.export__dictionary(bin_boundaries,
                             benchmark_filename,
                             header = ['element', 'boundaries'])


# Save the rate tables as a benchmark file
(rate_table, success) = mlw.combine__rate_tables(rate_tables)
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_rate-tables.tab")
success = mlw.export__dataframe(rate_table, benchmark_filename)
