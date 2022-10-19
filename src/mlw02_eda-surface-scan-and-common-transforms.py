# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 07:40:22 2022

@author: Tim

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


# Open the bank-telemarketing dataset...
## Note: the bank-telemarketing has semi-colons as separators, despite the .csv extension
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)


# What would we like to know about the data when we first get started?
## What are the dimensions of the data (rows, cols)?
## How many positive and negative outcomes are there?
## What are the basic stats for the columns?
## What does the data look like (distributions)?
## What else?



## What are the dimensions of the data (rows, cols)?
nrows, ncols = working.shape
print('\nThe working data has ', nrows, ' rows and ', ncols, 'columns.')


## How many positive and negative outcomes are there?
print('\nThe target element is distributed as follows:')
print(working[TARGET_ELEMENT_NAME].value_counts())


## What are the basic stats for the columns?
### Try the following, one at a time
working.describe()
working.describe(include=['number'])
working.describe(include=['number']).T

### save the stats for the numeric elements as a data frame
### and view them in the Variable Explorer
stats_numeric = working.describe(include=['number']).T
print('\nStats about the numeric elements:')
print(stats_numeric)

### YOUR TURN: do the same for the 'object' elements
### Are there any differences in the stats, compared to the numeric elements?
stats_object = working.describe(include=['object']).T
print('\nStats about the object elements:')
print(stats_object)

### Save the numeric stats to a benchmark file
BENCHMARK_FOLDER_NAME = "../bmrk/"
BENCHMARK_SEPARATOR   = "\t"
stats_numeric_filename = BENCHMARK_FOLDER_NAME + STUDY_NAME + "_stats_numeric.tab"
stats_numeric.to_csv(stats_numeric_filename, sep=BENCHMARK_SEPARATOR, index=True)

### Why bother saving the stats to a benchmark file?

### YOUR TURN: do the same for the 'object' elements
### Any changes you would make?

### INTERACTIVE: write a *function* to produce all of the stats and write
### them to benchmark file(s)...


## What does the data look like (distributions)?
### This is our first look at matplotlib.
### Let's start with a numeric element, age, and produce a histogram.
### See https://www.statology.org/pandas-histogram/
element_name = 'age'
working[element_name].plot.hist(bins=20)
title_string = 'What does the ' + element_name + ' distribution look like?'
plt.title(title_string)
plt.xlabel(element_name)
plt.ylabel('Frequency')
plt.show()

### YOUR TURN: By hand, repeat the above for a couple more numeric elements (your choice)

### Suppose we want to produce histograms for ALL of the numeric elements...
### This is our first use of a *for* loop
numerics = working.select_dtypes(include='number').columns.tolist()    # list of numeric elements
for element_name in numerics:
    print(element_name)
    working[element_name].plot.hist(bins=20)
    title_string = 'What does the ' + element_name + ' distribution look like?'
    plt.title(title_string)
    plt.xlabel(element_name)
    plt.ylabel('Frequency')
    plt.show()

### What do you notice about the age distribution?
### What do you notice about the duration distribution?
### Any other interesting notes???

### Do a log transform of duration and plot the distribution
### This is our first use of numpy
working['duration_log10'] = np.log10(working[working['duration']>0].duration)

### What do you notice about the duration_log10 distribution?
### Are there any other elements where you might apply a log?


### Next, let's look at a category (object) element, job, and produce a bar chart
### See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html
element_name = 'job'
working[element_name].value_counts().plot.barh().invert_yaxis()
title_string = 'What does the ' + element_name + ' distribution look like?'
plt.title(title_string)
plt.ylabel(element_name)
plt.xlabel('Frequency')
plt.show()

### What do you notice about the job distribution?

## YOUR TURN: produce bar charts for ALL of the object elements...

### YOUR TURN: write function(s) to produce ALL of the histograms
### and ALL of the bar charts...





### You can change many parameters to make the charts look more professional
### Let's look at bins and figsize for the histograms
element_name = 'age'
working[element_name].plot.hist(bins=25, figsize=(20,12))
title_string = 'What does the ' + element_name + ' distribution look like?'
plt.title(title_string)
plt.xlabel(element_name)
plt.ylabel('Frequency')
plt.show()



