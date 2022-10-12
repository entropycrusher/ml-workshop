# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:47:25 2022

@author: tgrae
"""

welcome_text = 'Welcome to the Machine Learning Workshop!'
print(welcome_text)

your_name = 'Tim'
print('\n' + 'Hi ' + your_name + '!! ' + welcome_text)

# Loading data (dgk_train.csv)
## Note that the following are constants, denoted by ALL CAPS
DATA_FILE_NAME   = "bank-telemarketing_train.csv"
DATA_FOLDER_NAME = "../data/"

import pandas as pd
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME)

# Your turn, do the same for the census-income dataset...
DATA_FILE_NAME   = "census-income_train.csv"
ELEMENT_NAMES    = ["age", "work", "fnlw", "edu", "edun", "marit", "occ", "rel", "race", "sex", "capg", "capl", "hours", "nativ", "ovr50"]
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, header=None, names=ELEMENT_NAMES)

# Your turn, do the same for the bank-telemarketing dataset...
DATA_FILE_NAME   = "bank-telemarketing_train.csv"
FILE_SEPARATOR   = ";"
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)
