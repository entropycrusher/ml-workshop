# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 08:54:23 2022

Machine Learning Workshop Session 8: Putting It All Together: EDA

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

TREE_BIN_CRITERION = 'entropy'
TREE_BIN_DEPTH     = 2
TREE_BIN_MIN_LEAF  = 1000
TREE_PLOT          = True

QUANTIZED_SUFFIX = "_q"
RATE_SUFFIX = "_r"
P_VALUE = 0.0001     # Note that we are NOT using .05 as our p-value.
BENCHMARK_FOLDER = "../bmrk/"


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
bin_boundaries = mlw.compute__bin_boundaries_for_all_numeric(predictors_train, target_train,
                                           criterion=TREE_BIN_CRITERION,
                                           depth    =TREE_BIN_DEPTH,
                                           min_leaf =TREE_BIN_MIN_LEAF,
                                           tree_plot=TREE_PLOT
                                           )

# Cycle thru the bin-able elements and apply the bin boundaries
predictors_train = mlw.apply__bin_boundaries_to_all_numeric(predictors_train, bin_boundaries,
                                           suffix=QUANTIZED_SUFFIX
                                           )
# Compute the target rate
target_rate = mlw.compute__target_rate(target_train)

# Estimate rates for all of the category elements
rate_tables = mlw.compute__rate_tables_for_all_category(predictors_train, target_train, target_rate,
                                           target_p_value           =P_VALUE,
                                           confidence_limits_p_value=P_VALUE
                                           )

# if any, cycle thru the rate-able elements and apply the rate tables
predictors_train = mlw.apply__rate_tables_to_all_category(predictors_train, rate_tables, target_rate,
                                    suffix=RATE_SUFFIX
                                    )

# Save the bin boundaries as a benchmark file
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_bin-boundaries.tab")
success = mlw.export__dictionary(bin_boundaries,
                             benchmark_filename,
                             header = ['element', 'boundaries'])

# Save the rate tables as a benchmark file
(rate_table, success) = mlw.combine__rate_tables(rate_tables)
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_rate-tables.tab")
success = mlw.export__dataframe(rate_table, benchmark_filename)
