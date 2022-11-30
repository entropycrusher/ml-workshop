# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 06:41:23 2022

Machine Learning Workshop Session 8: Putting It All Together: Transformation

@author: Tim G.

"""

# Import packages
## You can add more packages as you need them.
import pandas as pd
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

# Compute the target rate
target_rate = mlw.compute__target_rate(target_train)

# If any numeric elements exist, compute "smart" bin boundaries for them.
TREE_BIN_CRITERION = 'entropy'
TREE_BIN_DEPTH     = 2
TREE_BIN_MIN_LEAF  = 1000
bin_boundaries = mlw.compute__bin_boundaries_for_all_numeric(predictors_train, target_train,
                                           criterion=TREE_BIN_CRITERION,
                                           depth    =TREE_BIN_DEPTH,
                                           min_leaf =TREE_BIN_MIN_LEAF
                                           )

# Cycle thru the bin-able elements and apply the bin boundaries
QUANTIZED_SUFFIX = "_q"
predictors_train = mlw.apply__bin_boundaries_to_all_numeric(predictors_train, bin_boundaries,
                                           suffix=QUANTIZED_SUFFIX
                                           )

# Estimate rates for all of the category elements
P_VALUE = 0.0001
rate_tables = mlw.compute__rate_tables_for_all_category(predictors_train, target_train, target_rate,
                                           target_p_value           =P_VALUE,
                                           confidence_limits_p_value=P_VALUE
                                           )

# if any, cycle thru the rate-able elements and apply the rate tables
RATE_SUFFIX = "_r"
predictors_train = mlw.apply__rate_tables_to_all_category(predictors_train, rate_tables, target_rate,
                                    suffix=RATE_SUFFIX
                                    )

