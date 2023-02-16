# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 07:46:23 2023

Machine Learning Workshop: machine learning workshop 10: building models

@author: Tim G.

"""

# Import packages
## You can add more packages as you need them.
import pandas as pd
import mlw


# Define constants and configuration information.
STUDY_NAME          = "bank-telemarketing"
DATA_FOLDER_NAME    = "../data/"
FIGS_FOLDER         = "../figs/"
DATA_FILE_NAME      = "bank-telemarketing_train.csv"
FILE_SEPARATOR      = ";"

element_names       = [] # fill this list if names are *NOT* present in a header row
RENAME_ELEMENTS     = {} # {'from':'to'}
IGNORE_ELEMENTS     = []

BINARY_TARGET_NAME  = 'subscribed'

NUMERIC_NOMINAL_ELEMENTS = []
MISSING_VALUE_CHARACTER  = '.'

QUANTIZED_SUFFIX = "_q"
RATE_SUFFIX      = "_r"
BENCHMARK_FOLDER = "../bmrk/"
P_VALUE          = 0.0001     # Note that we are NOT using .05 as our p-value.



# YOUR TURN: Change the configuration information for your dataset.
## NOTE: Throughout this script, ALL_CAPS signifies configuration information.
## You probably will want to move *all* of the configuration info to the top
## of your script, eventually.


# Read the dataset.
if len(element_names) > 0:
    working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR,
                        header=None, names=element_names
                        )
else:
    working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)


# Rename elements if needed
if len(RENAME_ELEMENTS) > 0:
    working = working.rename(columns=RENAME_ELEMENTS)

# Ignore elements if needed
if len(IGNORE_ELEMENTS) > 0:
    working.drop(columns=IGNORE_ELEMENTS, inplace=True)


# Handle numeric nominal elements, if needed
for element in NUMERIC_NOMINAL_ELEMENTS:
    working[element] = working[element].astype(str)
    working[element].replace('nan', MISSING_VALUE_CHARACTER, inplace=True)

# Convert any object elements (string, nominal) to categorical, 
# add a '.' category for missing and replace any NaNs with '.'
success = mlw.convert__object_to_category(working)


# Read the saved bin boundaries from the model run
benchmark_filename = (BENCHMARK_FOLDER + STUDY_NAME + "_bin-boundaries.tab")
bin_boundaries = pd.read_csv(benchmark_filename, index_col=0, sep='\t')

dt = pd.read_csv(benchmark_filename, sep='\t', index_col=0, skiprows=1).T.to_dict()

dt = pd.read_csv(benchmark_filename, dtype={element_name : str}, sep='\t')
dt.set_index(element_name, inplace=True)


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


#### start of workshop 9 #######################################################

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

    scored_data_filename = BENCHMARK_FOLDER + STUDY_NAME + "_scores-train.tab"
    success = mlw.export__dataframe(pd.DataFrame(zip(estimate_train, target_train),
                                                 columns=[ESTIMATE_NAME, BINARY_TARGET_NAME]), 
                                    scored_data_filename)








