# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 07:40:22 2022

@author: tgrae
"""

# Start w/ an empty window

# Spyder: 
## code window, Variable Explorer, Plots, Console

# Windows Explorer
## ml-workshop folder, note the sub-folders: should see data and docs
## add bmrk, figs, rpts, and src

# Spyder: Open Preferences, mod the folders for Run and for Working directory
## look at Appearance: modify theme, fonts, etc. here
## restart the kernel for the preferences to take effect


welcome_text = 'Welcome to the Machine Learning Workshop!'
print(welcome_text)

your_name = 'Tim'
print('\n' + 'Hi ' + your_name + '!! ' + welcome_text)

# GitHub Desktop: the vault
## Save this file and look at GitHub Desktop.
## Commit the file at this point, and check back later after more additions

# Notebook: recording your thoughts and decisions
## Share my course notebook (in LaTeX), add a line or two and view in GitHub.

# Loading data (dgk_train.csv)
## Note that the following are constants, denoted by ALL CAPS
DATA_FILE_NAME   = "dgk_train.csv"
DATA_FOLDER_NAME = "../data/"


import pandas as pd
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME)

# Your turn, do the same for the census-income dataset...
## census-income has no header row with the column names
DATA_FILE_NAME   = "census-income_train.csv"
ELEMENT_NAMES    = ["age", "work", "fnlw", "edu", "edun", "marit", "occ", "rel", "race", "sex", "capg", "capl", "hours", "nativ", "ovr50"]
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, header=None, names=ELEMENT_NAMES)

# Your turn, do the same for the bank-telemarketing dataset...
## bank-telemarketing has semi-colons as separators, despite the .csv extension
DATA_FILE_NAME   = "bank-telemarketing_train.csv"
FILE_SEPARATOR   = ";"
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)





# Selecting, adding, removing, renaming, converting (object -> categorical),
## and the importance of working with lists
## Re-load the census-income dataset for the following...
DATA_FILE_NAME   = "census-income_train.csv"
ELEMENT_NAMES    = ["age", "work", "fnlw", "edu", "edun", "marit", "occ", "rel", "race", "sex", "capg", "capl", "hours", "nativ", "ovr50"]
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, header=None, names=ELEMENT_NAMES)

print(working.columns)
print(working.dtypes)

# How do you select a single element, say, for viewing?  
## Specify the element name in quotes or use the dot notation
print(working['edu'])
# Or
print(working.edu)

# How do you select multiple elements?  
## Specify the list of elements within brackets ...
print(working[['edu','occ','race']])

#   or create the list of elements first, then pass to the dataframe
ELEMENT_NAMES = ['edu','capg','age']
print(working[ELEMENT_NAMES])



# How do you select elements based on their 'type'?  Use the select_dtypes() method.  
print(working.select_dtypes(include='object'))			# Object elements
print(working.select_dtypes(include='number'))			# Number elements

# How do you obtain just the names of the elements, say, as a list?
nominals = working.select_dtypes(include='object').columns.tolist()    # Object elements
numerics = working.select_dtypes(include='number').columns.tolist()    # Number elements

# And use those lists to select portions of the data frame
## This is an example of working with lists of elements starts to be an advantage
## since we often want to apply different treatments to different groups (lists)
## of elements.
print(working[nominals])
print(working[numerics])



# How do I rename an element(s)?  
## Use the rename() method with a dictionary specifying the old name(s) 
## and the corresponding new name(s).
working = working.rename(columns={'capg':'capital_gain','capl':'captial_loss'})
#	or
working.rename(columns={'edu':'education','edun':'education_years'},inplace=True)








