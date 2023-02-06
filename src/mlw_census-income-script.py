# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:45:23 2023

Machine Learning Workshop: census-income script

@author: Tim G.

"""

# Import packages
## You can add more packages as you need them.
import pandas as pd
import mlw


# Define constants and configuration information.
STUDY_NAME          = "census-income"
DATA_FOLDER_NAME    = "../data/"
DATA_FILE_NAME      = "census-income_train.csv"
FILE_SEPARATOR      = ","

# YOUR TURN: Change the configuration information for your dataset.
## NOTE: Throughout this script, ALL_CAPS signifies configuration information.
## You probably will want to move *all* of the configuration info to the top
## of your script, eventually.


# Read the dataset.
element_names = ["age", "work", "fnlw", "edu", "edun", "marit", "occ", "rel", 
                 "race", "sex", "capg", "capl", "hours", "nativ", "ovr50"]    # fill this list if names are *NOT* present in a header row
if len(element_names) > 0:
    working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR,
                        header=None, names=element_names
                        )
else:
    working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)

# YOUR TURN: Read your dataset and look at it in Spyder.
## Did you read the dataset successfully?  How do you know?
## What do you know about your dataset thus far?




# Are there any elements that you want to rename?
## RENAME_ELEMENTS = {'from':'to'}
#RENAME_ELEMENTS = {'job':'occupation'}
#if len(RENAME_ELEMENTS) > 0:
#    working = working.rename(columns=RENAME_ELEMENTS)

# YOUR TURN: Rename any elements as you prefer.
## Are there any characters you can or should *NOT* use in element names?




# Are there any elements that you want to ignore?
#IGNORE_ELEMENTS = ["duration"]
#if len(IGNORE_ELEMENTS) > 0:
#    working.drop(columns=IGNORE_ELEMENTS, inplace=True)

# YOUR TURN: Specify any elements you want to ignore.
## Note: this list may change during the course of your project.




# Is the target element binary?  Do you need to "map" it to 0/1?
TARGET_ELEMENT_NAME = 'ovr50'
TARGET_MAPPING      = {' >50K':1, ' <=50K':0}
BINARY_TARGET_NAME  = 'ovr50k'

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
## They look like numbers, but it doesn't make sense to apply math operations to them.
NUMERIC_NOMINAL_ELEMENTS = []
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

#### From workshop 8 ###########################################################     

TREE_BIN_CRITERION = 'entropy'
TREE_BIN_DEPTH     = 2
TREE_BIN_MIN_LEAF  = 1000
TREE_PLOT          = True

QUANTIZED_SUFFIX = "_q"
RATE_SUFFIX      = "_r"
BENCHMARK_FOLDER = "../bmrk/"
P_VALUE          = 0.0001     # Note that we are NOT using .05 as our p-value.


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

#### end of workshop 8 #########################################################


#### new for workshop 9 ########################################################

# Save the bin boundaries as a benchmark file
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_bin-boundaries.tab")
success = mlw.export__dictionary(bin_boundaries,
                             benchmark_filename,
                             header = ['element', 'boundaries'])

# Save the rate tables as a benchmark file
(rate_table, success) = mlw.combine__rate_tables(rate_tables)
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_rate-tables.tab")
success = mlw.export__dataframe(rate_table, benchmark_filename)

# YOUR TURN: Produce these benchmark files for your dataset.
## Why might you want to save these files?




# Set the location for saving figures (plots, etc.)
FIGS_FOLDER = "../figs/"

# Produce the plot of what matters for the dataset
plot_filename = (FIGS_FOLDER + STUDY_NAME + "_what-matters-chart.png")
success = mlw.plot__what_matters_single_categories(rate_tables, 
                                                   target_rate,
                                                   STUDY_NAME,
                                                   P_VALUE,
                                                   filename=plot_filename
                                                   )
# YOUR TURN: Plot what matters for your dataset.
## Look at your plot and tell us what you see...
## Which elements are most predictive of a positive outcome?
## Of a negative outcome?
## Do any of the results surprise you?

# CHALLENGE:
## Can you use the plot__what_matters function to produce a visual profile
##  for a single element?




# Compute the uncertainty and complexity for each element
(uc_table, success) = mlw.compute__complexity_and_uncertainty(rate_tables)

# Plot the Uncertainty-Complexity (UC) Chart
plot_filename = (FIGS_FOLDER + STUDY_NAME + "_uc-chart.png")
success = mlw.plot__uc_chart(uc_table, target_rate, STUDY_NAME, QUANTIZED_SUFFIX,
                             filename=plot_filename
                             )
# YOUR TURN: Produce the UC chart for your dataset.
## Look at your plot and tell us what you see...
## Which elements reduce uncertainty the most?
## Which elements are the most/least complex?
## Do the most complex elements reduce uncertainty the most?
## Any surprises?
## What makes the UC chart different from the What Matters chart?
## How many categories were tested for significance?


# Save the UC results as a benchmark file
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_uc-table.tab")
success = mlw.export__dataframe(uc_table, benchmark_filename)
# YOUR TURN: Save the benchmark results.
## What do you notice about the names of the benchmark files that you've saved?




#### end of workshop 9 #########################################################

#### start of workshop 10 ######################################################

# train a panel-of-experts model, aka, a rate model
CANDIDATE_UNCERTAINTY_LIMIT = 1.0
ESTIMATE_NAME               ='estimate_train'
SCORETILES_BIN_DEPTH        = 4


# use a *backward-stepwise* procedure to produce a logistic regression model
## from the full list of candidates
(predictors_train, logistic_regression_model, success) = mlw.compute__logistic_regression(
                                                            predictors_train,
                                                            target_train,
                                                            uc_table,
                                                            RATE_SUFFIX,
                                                            P_VALUE,
                                                            uncertainty_limit=CANDIDATE_UNCERTAINTY_LIMIT
                                                            )

# YOUR TURN: Produce the panel-of-experts model for your dataset.
## Are you familiar with the idea of a backward-stepwise procedure?
## How does it work?  Did you notice any elements being removed?
## Were any elements deemed useless?  Why?
## Why use it?
## What other options are there?



# if a model is successfully produced...
if success:
    ## display the logistic diagnostics
    mlw.display__logistic_diagnostics(logistic_regression_model)

    ## produce the predictions for the training data using the fitted model
    (predictors_train, estimate_train, success) = mlw.apply__logistic_regression(
                                                    predictors_train,
                                                    logistic_regression_model
                                                    )
    estimate_train.name = ESTIMATE_NAME  # give the estimates Series a name


    # plot the ROC curve and add the area under the curve to the archive
    plot_filename = FIGS_FOLDER + STUDY_NAME + "_train-roc-chart.png"
    roc_area_train = mlw.plot__roc_curve(target_train, estimate_train,
                                         STUDY_NAME,
                                         filename=plot_filename
                                         )

    # plot the gain chart for the predictions and capture the scoretile boundaries and gain table
    plot_filename = FIGS_FOLDER + STUDY_NAME + "_train-gain-chart.png"
    (scoretile_boundaries, scoretile_train_table, success) = mlw.plot__gain_chart(
                                                                target_train, 
                                                                estimate_train, 
                                                                target_rate, 
                                                                STUDY_NAME,
                                                                bin_depth=SCORETILES_BIN_DEPTH,
                                                                filename=plot_filename                                                                                          
                                                                )

    top_scoretile_train = scoretile_train_table['rate'][-1]
    top_scoretile_gain_train = scoretile_train_table['rate'][-1]/target_rate
    
    benchmark_filename = BENCHMARK_FOLDER + STUDY_NAME + "_scoretile-train-table.tab"
    success = mlw.export__dataframe(scoretile_train_table, benchmark_filename)

# YOUR TURN: Produce the diagnostics, the estimates, the plots,
## and the benchmarks for your dataset.
## What do you notice about the diagnostics for your model?
## Which of your experts are weighted the most? the least?
## How do those weights track with what matters and with the UC chart?
## Are there any elements missing from the model that you expected to see?

## Any questions about the ROC chart (that we've seen previously)?
## Are you familiar with the Gain chart?
## What does it tell you?
## Do we need both?  Why/not?

#### end of workshop 10 ########################################################

