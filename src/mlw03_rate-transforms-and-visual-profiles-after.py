# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 07:40:22 2022

Machine Learning Workshop Session 3

@author: Tim G.

"""

# import packages
import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

# define constants and configuration information
STUDY_NAME          = "bank-telemarketing"
DATA_FOLDER_NAME    = "../data/"
DATA_FILE_NAME      = "bank-telemarketing_train.csv"
FILE_SEPARATOR      = ";"
TARGET_ELEMENT_NAME = 'y'
TARGET_MAPPING      = {'yes':1, 'no':0}  # this is a dictionary
BINARY_TARGET_NAME  = 'subscribed'       # this is the name for the new target

# Open the bank-telemarketing dataset...
## Note: the bank-telemarketing has semi-colons as separators, despite the .csv extension
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)


# How might you know or find out what makes an element a good predictor?
## What does it mean to be a good predictor?
## Can we figure out how a predictor "works"?

# First, let's create a new target that is 1/0 rather than yes/no.
## We define the TARGET_MAPPING above (from:to) from yes/no to 1/0.
## Then, we can use the 'map' method to apply our mapping to the target element
## to create a new target.
working[BINARY_TARGET_NAME] = working[TARGET_ELEMENT_NAME].map(TARGET_MAPPING).astype('float')

### YOUR TURN: Find out if the mapping worked...




## Here's another way you can check your work: using a crosstab
pd.crosstab(working[TARGET_ELEMENT_NAME], working[BINARY_TARGET_NAME])


# How would we know if, say, a particular 'job' category is a good predictor
## that someone is likely to 'subscribe'?
## Maybe a job category has a high subscription rate (probability)...

## What does it mean to have a high rate or a low rate?
## Always ask, "Compared to what?"
## What is our baseline for comparison?

## How can we compute the overall rate of positive outcomes?
working[BINARY_TARGET_NAME].mean()

## YOUR TURN: Check if this calculation is right...




# Let's calculate the rates of positive outcomes for each category of 'job'...

## First, what are the categories and how frequently do they occur?
working['job'].unique()
working['job'].value_counts()

## We could use a crosstab to get an idea of the rates...
pd.crosstab(working['job'], working[BINARY_TARGET_NAME])

## Better yet, we can save the crosstab as a dataframe,
## then calculate the total number of rows and rate for each category.
crosstab_vs_outcome = pd.crosstab(working['job'], working[BINARY_TARGET_NAME])
crosstab_vs_outcome['total'] = crosstab_vs_outcome[0.0] + crosstab_vs_outcome[1.0]
crosstab_vs_outcome['rate']  = crosstab_vs_outcome[1.0] / crosstab_vs_outcome['total']

## And we'd much rather have the categories sorted...
### ALWAYS SORT EVERYTHING YOU PRESENT!!!
crosstab_vs_outcome.sort_values('rate',ascending=False)
crosstab_vs_outcome = crosstab_vs_outcome.sort_values('rate',ascending=False)

## What do these results tell you?  How can you use these results... today?

## YOUR TURN: write a function to return the rates for a specified element,
## and a specified (binary) target element.  (You might use more than one.)
## Then run your function for all of the category elements.
## Look thru your results.  What stands out to you?
def compute__rate_crosstab(df, 
                           target_element_name, 
                           element_name):
    '''  
    Parameters
    ----------
    df : dataframe
        the working dataframe.
    target_element_name : string
        the name of the (binary) target element.
    element_name : string
        the name of the (category) element of interest.

    Returns
    -------
    crosstab_table : dataframe
        the crosstab for the element versus the target, with the addition
        of the total count and the rate for each category
    '''
    crosstab_table          = pd.crosstab(df[element_name], df[target_element_name])
    crosstab_table['total'] = crosstab_table[0.0] + crosstab_table[1.0]
    crosstab_table['rate']  = crosstab_table[1.0]/crosstab_table['total']
    crosstab_table          = crosstab_table.sort_values('rate',ascending=False)
    
    return(crosstab_table)

print(compute__rate_crosstab(working, BINARY_TARGET_NAME, 'education'))

object_elements = working.select_dtypes(include='object').columns.tolist()

for item in object_elements:
    print('\n' + item )
    print(compute__rate_crosstab(working, BINARY_TARGET_NAME, item))



# This is all good, but wouldn't you rather look at a chart?
## How do we plot the rates for each element?
## What would we like to see on the plot?

## Start with our usual, horiz bar chart
crosstab_vs_outcome['rate'].plot.barh().invert_yaxis()
title_string = 'What are the category rates for job?'
plt.title(title_string)
plt.ylabel('job')
plt.xlabel('Frequency')
plt.show()

### Compared to what?  What is our reference value?
target_rate = working[BINARY_TARGET_NAME].mean()
TARGET_RATE_LINE_COLOR          = 'tab:gray' # 'tab:gray' is visible but unobtrusive
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 0.5

crosstab_vs_outcome['rate'].plot.barh().invert_yaxis()
plt.axvline(target_rate, color=TARGET_RATE_LINE_COLOR)
title_string = 'Visual profile for job'
plt.title(title_string)
plt.ylabel('job')
plt.xlabel('Frequency')
plt.show()

## YOUR TURN: write a function to produce the *visual profile* chart for 
## a specified element, then run for all of the category elements




## What might you like to do to improve the quality and/or information
## shown on the visual profile chart?
### Tableau demo and the What Matters chart


# How can we produce similar results for numeric elements?
## Use binning (aka, bucketing or quantizing) to cut the numbers into ranges
BIN_COUNT = 4
(out_ignore, bin_boundaries) = pd.qcut(working['age'], BIN_COUNT,
                                       retbins=True,
                                       duplicates='drop')

## Remember this: Everything you do has to work on new or different data!
## Refine the first and last boundaries to handle data beyond the range
bin_boundaries[0]                     = -np.inf
bin_boundaries[len(bin_boundaries)-1] =  np.inf

## Add an element in the working file for 'quantized age'
### Note that here you're *applying* the bin boundaries that you already created above.
### Note the naming convention for the 'quantized' element
working['age_q'] = pd.cut(working['age'], bin_boundaries,
                          include_lowest = True,
                          duplicates     ='drop',
                          right          = True
                          )

## Did it work?
working['age_q'].value_counts()  # we expect roughly equal counts in each bin


## YOUR TURN: write a function to add a binned element to the working dataset
### and return the bin boundaries for a specified (numeric) element and 
### a specified number of bins.  Then run for ALL of the numeric elements.

### How many bins should you use?



## CHALLENGE: Produce a visual profile for a (binned) numeric element.

## EXTRA CHALLENGE: Write a function to produce visual profiles for ALL elements.


