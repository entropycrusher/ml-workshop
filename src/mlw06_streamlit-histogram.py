# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:22:54 2022

@author: tgrae
"""

import streamlit         as st
import pandas            as pd
import matplotlib.pyplot as plt


# define constants and configuration information
STUDY_NAME          = "bank-telemarketing"
DATA_FOLDER_NAME    = "../data/"
DATA_FILE_NAME      = "bank-telemarketing_train.csv"
FILE_SEPARATOR      = ";"



# Open the bank-telemarketing dataset...
working = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, sep=FILE_SEPARATOR)

# Select the numeric elements and put them in a list
candidates = working.select_dtypes(include='number').columns.tolist()

if(True):
    # Put a title on the page
    st.title('Exploratory Data Analysis')


if(False):
    # Add a selectbox for choosing an element to plot:
    element = st.selectbox('Which element would you like to view?', candidates)

if(False):
    st.write(element)   # display the name of the selected element

if(False):
    # Compute and display the standard stats for the selected element
    st.subheader("Standard Statistics for " + element)
    st.table(working[element].describe())

if(False):
    # Display a histogram for the selected element
    st.subheader("Histogram for " + element)
    fig, ax = plt.subplots()
    ax.hist(working[element])
    plt.xlabel(element)
    plt.ylabel("Count")
    st.pyplot(fig)



if(False):
    # Add a slider to the sidebar to set the number of bins for the histogram:
    bins_count = st.sidebar.slider(
        label='How many bins for the histogram?',
        min_value=2, max_value=50, step=1, value = 25, format="%d"
    )

    # Display a histogram for the selected element
    st.subheader("Histogram for " + element)
    fig, ax = plt.subplots()
    ax.hist(working[element], bins=bins_count)
    plt.xlabel(element)
    plt.ylabel("Count")
    st.pyplot(fig)




if(False):
    # Move the selectbox to the sidebar
    element = st.sidebar.selectbox('Which element would you like to view?', candidates)




