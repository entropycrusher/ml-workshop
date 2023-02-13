# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:22:54 2023

@author: tgrae
"""

import streamlit         as st
import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

# define constants and configuration information
BENCHMARK_FOLDER      = '../bmrk/'
STUDY_NAME            = 'bank-telemarketing'
TARGET_ELEMENT_NAME   = "subscribed"
SCORES_TRAIN_SUFFIX   = "_scores-train.tab"
scores_train_filename = (STUDY_NAME + SCORES_TRAIN_SUFFIX)

# load the scores and outcomes from the training data
scores_train = pd.read_csv(BENCHMARK_FOLDER + scores_train_filename, sep='\t')
scores_train = scores_train.sort_values(by='estimate_train', ascending=False)


# compute the target rate (observed rate of positive outcomes)
## and get the number of rows in the training data
target_rate             = scores_train[TARGET_ELEMENT_NAME].mean()
number_of_model_records = len(scores_train)


# Put a title on the page
st.title('Direct Marketing Profit Simulator')

# Add a slider to the sidebar to set the number of prosepcts:
PROSPECTS_MIN        =   1000
PROSPECTS_MAX        = 100000
PROSPECTS_NUM_POINTS =    100
PROSPECTS_STEP       = int(PROSPECTS_MAX/PROSPECTS_NUM_POINTS)

number_of_prospects = st.sidebar.slider(
                        label='How many prospects for the campaign?',
                        min_value=PROSPECTS_MIN, max_value=PROSPECTS_MAX, 
                        step=PROSPECTS_STEP, value = PROSPECTS_MAX, format="%d"
                        )

prospects_delta = number_of_prospects/PROSPECTS_NUM_POINTS

# Add a slider to the sidebar to set the value per response:
value_per_response  = st.sidebar.slider(
                        label='What is the average value of a response?',
                        min_value=1.0, max_value=25.0, step=0.25, value = 10.00, format="%f"
                        )

# Add a slider to the sidebar to set the cost per contact:
cost_per_contact    = st.sidebar.slider(
                        label='What is the average cost per contact?',
                        min_value=.10, max_value=5.00, step=0.10, value = 1.00, format="%f"
                        )


# initialize a dataframe to capture all of the simulation results
simulator_table = pd.DataFrame(columns=['number_of_contacts',
                                        'responders',
                                        'response_rate',
                                        'revenue',
                                        'cost',
                                        'profit',
                                        'roi'
                                        ])

# add the "zero" row to the table
new_row = {'number_of_contacts' : 0.0,
           'responders'         : 0.0,
           'response_rate'      : 0.0,
           'revenue'            : 0.0,
           'cost'               : 0.0,
           'profit'             : 0.0,
           'roi'                : np.nan
           }
simulator_table = pd.concat([simulator_table, pd.DataFrame([new_row])], ignore_index=True)


# compute the model outcomes across a range for the number of contacts
contact_counts = pd.Series(np.linspace(int(prospects_delta), int(number_of_prospects), int(PROSPECTS_NUM_POINTS)))
for number_of_contacts_model in contact_counts:
    prospects_per_model_record = number_of_prospects/number_of_model_records
    depth_in_model_records     = round(number_of_model_records * number_of_contacts_model / number_of_prospects)
    positive_model_responses   = scores_train[TARGET_ELEMENT_NAME][:depth_in_model_records].sum()
    
    responders_model           = round(positive_model_responses * prospects_per_model_record)
    response_rate_model        = responders_model/number_of_contacts_model
    revenue_model              = value_per_response * responders_model
    cost_model                 = cost_per_contact   * number_of_contacts_model
    profit_model               = revenue_model - cost_model
    roi_model                  = profit_model  / cost_model
    #print(number_of_contacts_model, profit_model)

    # capture the results in a dataframe and create plots
    new_row = {'number_of_contacts' : int(number_of_contacts_model),
               'responders'         : responders_model,
               'response_rate'      : response_rate_model,
               'revenue'            : revenue_model,
               'cost'               : cost_model,
               'profit'             : profit_model,
               'roi'                : roi_model
               }
    simulator_table = pd.concat([simulator_table, pd.DataFrame([new_row])], ignore_index=True)


# define plot parameters
FIGURE_WIDTH        = 16
FIGURE_HEIGHT       = 16
CURVE_COLOR         = 'tab:blue'
COST_COLOR          = 'tab:red'
ZERO_COLOR          = 'tab:gray'
LINE_STYLE          = '--'
COST_STYLE          = ':'
LINE_WIDTH          = .75

# determine values for the 'No Model' situation
profit_all_no_model = number_of_prospects * (value_per_response * target_rate - cost_per_contact)
cost_all_no_model   = number_of_prospects * cost_per_contact
roi_no_model        = (value_per_response * target_rate - cost_per_contact)/cost_per_contact

# Initialize the subplot function using the desired number of rows and columns
figure, axis = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

# plot profit
axis[0, 0].plot(simulator_table['number_of_contacts'], simulator_table['profit'],
                color=CURVE_COLOR,
                lw=LINE_WIDTH)
axis[0, 0].plot([0, number_of_prospects], 
                [0, profit_all_no_model], 
                color=COST_COLOR, ls=COST_STYLE, lw=LINE_WIDTH)
axis[0, 0].axhline(0.0, color=ZERO_COLOR, ls=LINE_STYLE, lw=LINE_WIDTH)
axis[0, 0].set_title("Profit (Model and No Model) vs Number of Contacts")

# plot revenue and cost
axis[0, 1].fill_between(simulator_table['number_of_contacts'], 
                        simulator_table['revenue'],
                        simulator_table['cost'],
                        color='C1', alpha=0.1)
axis[0, 1].plot(simulator_table['number_of_contacts'], simulator_table['revenue'],
                color=CURVE_COLOR, lw=LINE_WIDTH)
axis[0, 1].plot(simulator_table['number_of_contacts'], simulator_table['cost'],
                color=COST_COLOR,  lw=LINE_WIDTH)
axis[0, 1].set_title("Revenue and Cost (Model) vs Number of Contacts")

# plot roi
axis[1, 0].plot(simulator_table['number_of_contacts'], simulator_table['roi'],
                color=CURVE_COLOR, lw=LINE_WIDTH)
axis[1, 0].axhline(roi_no_model, color=COST_COLOR, ls=COST_STYLE, lw=LINE_WIDTH)
axis[1, 0].axhline(0.0, color=ZERO_COLOR, ls=LINE_STYLE, lw=LINE_WIDTH)
axis[1, 0].set_title("ROI (Model and No Model) vs Number of Contacts")

# plot responders
axis[1, 1].plot(simulator_table['number_of_contacts'], simulator_table['responders'],
                color=CURVE_COLOR, lw=LINE_WIDTH)
axis[1, 1].plot([0, number_of_prospects], 
                [0, number_of_prospects * target_rate], 
                color=COST_COLOR, ls=COST_STYLE, lw=LINE_WIDTH)
axis[1, 1].set_title("Responders (Model and No Model) vs Number of Contacts")

# display the plots
st.pyplot(figure)

# CHALLENGE: display (say, in the sidebar) the number of contacts, revenue, 
## cost, responders and profit at max profit - and highlight that point on each chart

# CHALLENGE: add a budget constraint on the number of contacts

# CHALLENGE: add a revenue goal and/or a new customer goal
