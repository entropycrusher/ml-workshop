# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
FILE_SEPARATOR   = ";"
DATA_FILE_NAME   = "census-income_train.csv"
DATA_FOLDER_NAME = "../data/"
ELEMENT_NAMES    = ["age", "work", "fnlw", "edu", "edun", "marit", "occ", "rel", "race", "sex", "capg", "capl", "hours", "nativ", "ovr50"]



welcome_text = 'Welcome to the Machine Learning Workshop!'
print(welcome_text)

# Loading data (dgk_train.csv)
## Note that the following are constants, denoted by ALL CAPS

working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, header=None, names=ELEMENT_NAMES)
#working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME)

# Your turn, do the same for the census-income dataset...

# Your turn, do the same for the bank-telemarketing dataset...
## bank-telemarketing has semi-colons as separators, despite the .csv extension
DATA_FILE_NAME   = "bank-telemarketing_train.csv"
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)

