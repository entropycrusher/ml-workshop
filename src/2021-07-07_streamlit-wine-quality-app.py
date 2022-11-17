# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:06:02 2021

@author: Owner
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
     page_title="Wine Quality Classifier",
     page_icon=":penguin:",
     layout="wide",
     initial_sidebar_state="expanded",
     )


st.title('Wine Quality Classifier')

#st.markdown(' ## Designing a high-quality wine')
#st.markdown(' ### :point_left: Check the sidebar for more details!')
st.markdown(' ## Move the sliders to design your wine.')
#st.markdown(' ### (All variables are initially set to the mean values from the dataset.)')

break_line = '<hr style="border:2px solid gray"> </hr>'

st.sidebar.markdown("### Background")
st.sidebar.markdown("This project was built from almost 1,600 red wine samples.  \
    The idea is to learn what makes a high-quality wine beyond just the marketing mumbo jumbo on the bottle. \
        For more info, check out the dataset on [Kaggle](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).")
st.sidebar.markdown(break_line, unsafe_allow_html = True)
st.sidebar.markdown("### Examples to try")
st.sidebar.markdown("Try `Alcohol 12.0`, `Density 0.994`, `Volatile Acidity 0.3`, `Fixed Acidity 14.0`, `Chlorides 0.1`, `Sulfates 0.7`.  The result is High-Quality!  What would make it low-quality?  Make some changes, see what happens!")
st.sidebar.markdown(break_line, unsafe_allow_html = True)
st.sidebar.markdown("### Key to Sliders")
st.sidebar.markdown("Wine is a complex thing!\
                        Here are some hints for making high-quality wine...")
st.sidebar.markdown("* Alcohol: This is the percent alcohol content of the wine.\
                        The more, the better.")
st.sidebar.markdown("* Volatile Acidity: The amount of acetic acid in wine. Too much leads gives an unpleasant, vinegar taste.")
st.sidebar.image("photo-1606655762823-63f1cda5431f.jpg")
st.sidebar.markdown(break_line, unsafe_allow_html = True)
st.sidebar.markdown("### Authors")
st.sidebar.markdown("This project was built by Pete and Tim.")

# Add sliders for each of the elements in the model
alcohol          = st.slider(key='alc', label="Alcohol",          min_value=8.7,     max_value=14.0,     step=0.1,   value = 10.445,    help='Increase alcohol content to improve quality')
density          = st.slider(key='den', label="Density",          min_value=0.99007, max_value= 1.00369, step=0.001, value =  0.996699, help='Decrease density content to improve quality', format="%f")
volatile_acidity = st.slider(key='vol', label="Volatile Acidity", min_value=0.12,    max_value= 1.33,    step=0.001, value =  0.528506, help='Decrease volatile_acidity content to improve quality')
fixed_acidity    = st.slider(key='fix', label="Fixed Acidity",    min_value=4.6,     max_value=15.6,     step=0.05,  value =  8.28518,  help='Increase fixed_acidity content to improve quality')
chlorides        = st.slider(key='chl', label="Chlorides",        min_value=0.012,   max_value= 0.61,    step=0.001, value =  0.087738, help='Decrease chlorides content to improve quality')
sulphates        = st.slider(key='sul', label="Sulphates",        min_value=0.37,    max_value= 2.0,     step=0.1,   value =  0.664296, help='Increase sulphates content to improve quality')


# Compute the weighted sum of the factors based on the output of the logisitic regression
factor_sum = (
              alcohol          *   1.0869 + 
              density          * -15.5315 + 
              volatile_acidity *  -2.7270 + 
              fixed_acidity    *   0.3186 + 
              chlorides        * -19.1265 +
              sulphates        *   2.6034
              )

# Compute the logistic model score for the given slider settings and display it on the page
pred = 1/(1 + np.exp(-factor_sum))


#st.header('Is the wine you designed of high quality?')
#st.write(pred*100)

# Steps beyond the above...
# - Add commentary about the data elements and how the model was developed.
# - Improve the layout, maybe a side-by-side with sliders and prediction-plus-picture
# - Perhaps add an optimization component where there are costs associated with each element

if pred > 0.5:
    st.markdown("## Based on your settings, I'd say you're drinking... High-quality wine\n")
    #st.markdown("# High-quality wine\n")
    st.image('yummy-face-emoji.png', use_column_width=True)
else:
    st.markdown("## Based on your settings, I'd say you're drinking... Low-quality wine\n")
    #st.markdown("# Low-quality wine\n")
    st.image('yucky-face-image.gif', use_column_width=True)



