# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:57:55 2022

@author: Tim G.
"""

# Below are functions provided by Discovery Corps, Inc.
## for use in the Machine Learning Workshop.
## They are provided as learning tools, without warranty, 
## either expressed or implied.

import pandas                 as     pd
import numpy                  as     np
import matplotlib.pyplot      as     plt
import seaborn                as     sns              # another plotting package
from sklearn.metrics          import roc_auc_score    # for measuring performance
from sklearn.metrics          import roc_curve        # for plotting performance
from sklearn.model_selection  import train_test_split # for partitioning a dataset
from sklearn.tree             import DecisionTreeClassifier # for building a decision tree


ROC_FIGURE_WIDTH        = 16
ROC_FIGURE_HEIGHT       = 16
ROC_ASPECT_RATIO        = 'equal'
ROC_CURVE_COLOR         = 'tab:blue'
ROC_DIAGONAL_COLOR      = 'tab:red'
ROC_LINE_STYLE          = '--'
ROC_LINE_WIDTH          = .75




def plot__roc_curve(target, estimates,
                    study_name='bank-telemarketing'):
    '''
    Parameters
    ----------
    target : Series
        the target (observed) outcome column.
    estimates : Series
        the estimated outcome colum.
    study_name : string, optional
        the name of the study, used for titling the charts. 
        The default is 'bank-telemarketing'.

    Returns
    -------
    area_under_curve : float
        the area under the ROC curve.
        Can be used as a simple performance metric

    '''
    area_under_curve     = roc_auc_score(target, estimates)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(target, estimates)
    
    fig = plt.figure(figsize=(ROC_FIGURE_WIDTH, ROC_FIGURE_HEIGHT))
    ax = fig.add_subplot(111)
    ax.set_aspect(ROC_ASPECT_RATIO)
    plt.plot(false_positive_rate, true_positive_rate,
             color=ROC_CURVE_COLOR,
             lw=0.75,
             label='area = %0.4f' % area_under_curve)
    plt.plot([0, 1], [0, 1], color=ROC_DIAGONAL_COLOR, ls=ROC_LINE_STYLE, lw=ROC_LINE_WIDTH)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC chart for ' + study_name)
    sns.despine()
    plt.legend(loc="lower right")
    plt.show();
    
    return(area_under_curve)



def convert__object_to_category(df, missing_value_character='.'):
    '''
    Parameters
    ----------
    df : dataframe
        the dataframe containing elements to be converted.
    missing_value_character : string. optional
        the character to use when replacing missing values.  The default is '.'

    Returns
    -------
    success : Boolean
         a success flag.  The object elements of the dataframe 
         are converted to category elements in place.
    '''
    nominals = df.select_dtypes(include='object').columns.tolist()
    for element in nominals:
        df[element] = df[element].astype('category')
        if '.' not in df[element].cat.categories:
            df[element] = df[element].cat.add_categories('.').fillna('.')
        else:
            df[element] = df[element].fillna('.')

    return(True)



def split__dataset(dataset, target, test_set_fraction=0.5, random_seed=62362):
    '''
    Parameters
    ----------
    dataset : dataframe
        dataset to be split into training and testing subsets.
    target : string
        name of the target element.
    test_set_fraction : float, optional
        the fraction of rows to assign to the testing subset. The default is 0.5.
    random_seed : integer, optional
        the random seed to ensure repeatability, if desired. The default is 62362.

    Returns
    -------
    (input_train, input_test, target_train, target_test, success) : 
        (dataframe, dataframe, series, series, Boolean)
        tuple including the following:
            training predictor(s), the testing predictor(s),
            training outcome, testing outcome, and a success flag
        Note that all indexes for the training components will be reset and in sync
        as will the indexes for the testing components.
    '''
    # identify the predictors
    predictors = dataset.columns.to_list()
    predictors.remove(target)

    # use the scikit learn function to split the dataset
    # note that the target_series is used to stratify the dataset.
    (input_train, input_test, target_train, target_test) = train_test_split(
        dataset[predictors],
        dataset[target],
        stratify    =dataset[target],
        test_size   =test_set_fraction,
        random_state=random_seed
        )

    # sort the training input and target by index, then reset the indexes
    input_train  = input_train.sort_index(axis=0).copy()
    target_train = target_train.sort_index(axis=0).copy()
    input_train.reset_index( inplace=True, drop=True)
    target_train.reset_index(inplace=True, drop=True)

    # sort the testing  input and target by index, then reset the indexes
    input_test   = input_test.sort_index(axis=0).copy()
    target_test  = target_test.sort_index(axis=0).copy()
    input_test.reset_index( inplace=True, drop=True)
    target_test.reset_index(inplace=True, drop=True)

    # return the training and testing components plus a success flag
    return(input_train, input_test, target_train, target_test, True)





def run__standard_prep(data_folder_name="../data/",
                       data_file_name="train.csv",
                       file_separator=",",
                       study_name="study_name",
                       rename_elements={},
                       ignore_elements=[],
                       target_element_name='target',
                       target_mapping={},
                       binary_target_name='binary_target',
                       numeric_nominal_elements=[],
                       missing_value_character='.',
                       test_set_fraction=0.5,
                       random_seed_value=62362
                       ):
    '''
    Parameters
    ----------
    data_folder_name : string, optional
        location of the data file. The default is "../data/".
    data_file_name : string, optional
        name of the data file. The default is "train.csv".
    file_separator : string, optional
        the delimiter between fields in the data file. The default is ",".
    study_name : string, optional
        the name of the study (primarily for labeling purposes). The default is "study_name".
    rename_elements : dictionary, optional
        mapping dictionary from original names to new names. The default is {}.
    ignore_elements : list of strings, optional
        the names of the elements to ignore. The default is [].
    target_element_name : string, optional
        the name of the target element. The default is 'target'.
    target_mapping : dictionary, optional
        mapping dictionary from original labels to 0/1. The default is {}.
    binary_target_name : string, optional
        the name of the binary target element. The default is 'binary_target'.
    numeric_nominal_elements : list of strings, optional
        the names of the numeric elements to treat as nominal. The default is [].
    missing_value_character : string, optional
        single character to use for missing values. The default is '.'.
    test_set_fraction : float, optional
        the fraction of the dataset to reserve for testing. The default is 0.5.
    random_seed_value : int, optional
        the random seed for splitting data into training and testing. The default is 62362.

    Returns
    -------
    None.

    '''
    # Read the dataset.
    working = pd.read_csv(data_folder_name + data_file_name, sep=file_separator)
    
    # Rename elements
    if len(rename_elements) > 0:
        working = working.rename(columns=rename_elements)
    
    # Ignore elements
    if len(ignore_elements) > 0:
        working.drop(columns=ignore_elements, inplace=True)

    # Map target to 0/1
    if len(target_mapping) > 0:
        working[binary_target_name] = working[target_element_name].map(target_mapping).astype('float')
        working.drop(columns=target_element_name, inplace=True)
    else:
        binary_target_name = target_element_name

    # Drop rows where the target is missing
    working = working.dropna(subset=[binary_target_name])


    # Drop any columns that are useless
    ## First, identify any columns consisting entirely of NaNs and drop them
    all_nan = working.isna().all(axis=0)
    to_drop = all_nan[all_nan].index.to_list()
    if len(to_drop) > 0:
        print('\n\nThe following elements are entirely missing (NaN), and they will be dropped:', to_drop)
        working.drop(columns=to_drop, inplace=True)

    ## Next, identify any columns that are constant (single-valued) and drop them
    constant_elements = []
    for element in working.columns.to_list():
        if not (working[element] != working[element].iloc[0]).any():
            constant_elements.append(element)
    if len(constant_elements) > 0:
        print('\n\nThe following elements are constant, and they will be dropped:', constant_elements)
        working.drop(columns=constant_elements, inplace=True)

    # Convert numeric nominal elements
    for element in numeric_nominal_elements:
        working[element] = working[element].astype(str)
        working[element].replace('nan', missing_value_character, inplace=True)

    # Convert object elements (string, nominal) to categorical
    success = convert__object_to_category(working)

    # Split the working dataset into a training subset and testing subset
    (predictors_train, predictors_test, target_train, target_test, success) = split__dataset(
                    working, binary_target_name,
                    test_set_fraction = test_set_fraction,
                    random_seed       = random_seed_value
                    )

    return(working, 
           binary_target_name, 
           predictors_train, 
           predictors_test, 
           target_train, 
           target_test
           )



def compute__target_rate(target):
    '''
    Parameters
    ----------
    target : Series
        the target element.

    Returns
    -------
    (target_rate, success) : float, Boolean
        the fraction of positive outcomes and a success flag
    '''
    return target.mean()



def compute__bin_boundaries_for_all_numeric(predictors, target,
                                            criterion='entropy',
                                            depth=2,
                                            min_leaf=1000):
    '''
    Parameters
    ----------
    predictors : dataframe
        the collection of predictors.
    target : series
        the target outcome.
    criterion : string, optional
        the optimization criterion for tree building. The default is 'entropy'.
    depth : integer, optional
        the maximum depth of the tree. The default is 2.
    min_leaf : integer, optional
        the minimum number of observations at a leaf of the tree. The default is 1000.

    Returns
    -------
    (bin_boundaries, success) : dict, Boolean
        the dictionary of bin bounaries for the numeric elements plus a success flag
    '''
    # get the list of numeric elements
    numeric_element_names = list__numeric_elements(predictors)

    # create empty dict for storing the bin boundaries
    bin_boundaries = {}

    # if any, cycle thru the numeric elements and determine bin boundaries
    for element_name in numeric_element_names:
        # use the function to compute the tree bin boundaries
        tree_bin_boundaries = compute__tree_bin_boundaries(predictors[element_name], target,
                                      tree_bin_criterion=criterion,
                                      tree_bin_depth    =depth,
                                      tree_bin_min_leaf =min_leaf
                                      )

        # add the key-value pair for the element name and the bin_boundaries to the dict
        bin_boundaries[element_name] = tree_bin_boundaries

    return bin_boundaries




def apply__bin_boundaries_to_all_numeric(predictors, bin_boundaries, suffix='_q'):
    '''
    Parameters
    ----------
    predictors : dataframe
        the collection of predictors.
    bin_boundaries : dict
        the dictionary of bin bounaries for the numeric elements.
    suffix : string, optional
        the suffix used to form the names of the binned elements.  The default is '_q'.

    Returns
    -------
    predictors : dataframe
        the *updated* collection of predictors
    success : Boolean
        success flag
    '''

    # if the bin_boundaries dictionary is not empty, find any available elements to bin
    if bin_boundaries:
        elements_with_bin_boundaries = list(bin_boundaries.keys())
        all_elements = predictors.columns.to_list()
        elements_to_bin = list(set(all_elements).intersection(elements_with_bin_boundaries))

        # if there are elements in the dataset with corresponding bin boundaries...
        for element_name in elements_to_bin:
            # apply the bin boundaries and add the new element to the dataset
            category_element_name = element_name + suffix
            category_element = apply__bin_boundaries(predictors[element_name],
                                          bin_boundaries[element_name])
            predictors = pd.concat([predictors, category_element.rename(category_element_name)], axis=1)

    return predictors




def apply__bin_boundaries(element, bin_boundaries):
    '''
    Parameters
    ----------
    element : Series, numeric
        the values to be binned.
    bin_boundaries : list of floats
        the list of boundaries obtained from the binning process.

    Returns
    -------
    element_q : Series, category
        the binned (quantized) element (categorical).
    '''
    element_q = pd.cut(element, bin_boundaries,
                         include_lowest = True,
                         duplicates     ='drop',
                         right          = True
                         )
    
    # add a category for missing values, even if none are present.
    # if any are present, they will be filled with a 'dot' (period)
    element_q = element_q.cat.add_categories('.').fillna('.')

    return element_q




def list__numeric_elements(df):
    '''
    Parameters
    ----------
    df : pandas DataFrame
        typically the working DataFrame.

    Returns
    -------
    list
        the list of numeric elements that are present.
    '''
    return df.select_dtypes(include='number').columns.tolist()



def compute__tree_bin_boundaries(predictor_series, target_series,
                                 tree_bin_criterion='entropy',
                                 tree_bin_depth=2,
                                 tree_bin_min_leaf=1000
                                 ):
    '''
    Parameters
    ----------
    predictor_series : Series, numeric
        the predictor (input).
    target_series : Series, numeric, binary
        the target, or outcome of interest.
    tree_bin_criterion : string, optional
        the optimization criterion for tree building. 
        The default is 'entropy'.
    tree_bin_depth : int, optional
        the maximum depth of the tree. 
        The default is 2.
    tree_bin_min_leaf : int, optional
        the minimum number of observations at a leaf of the tree. 
        The default is 1000.

    Returns
    -------
    bin_boundaries : list, numeric
        the list of bin cut points (or bin boundaries) to be used by a
        downstream quantizer
    '''
    
    # bolt the predictor_series and the target together into a mini dataframe,
    #   and then drop any rows that contain NaNs in either column
    mini_df = pd.concat([predictor_series, target_series], axis=1)
    mini_df = mini_df.dropna()
    mini_df = mini_df.reset_index(drop=True)
    predictor_series = mini_df[predictor_series.name]
    target_series    = mini_df[target_series.name]

    # create a decision tree classifer object
    tbc = DecisionTreeClassifier(criterion         = tree_bin_criterion,
                                 max_depth         = tree_bin_depth,
                                 min_samples_split = 2*tree_bin_min_leaf,
                                 min_samples_leaf  = tree_bin_min_leaf)

    # train the decision tree classifer
    tbc = tbc.fit(pd.DataFrame(predictor_series),target_series)

    # assign a node id to each row of data
    node_id = pd.Series(tbc.apply(pd.DataFrame(predictor_series)))

    # extract the boundaries based on node_id
    bin_boundaries = list(predictor_series.groupby(node_id).max())

    # refine the first and last boundaries for data beyond the training range
    bin_boundaries.insert(0, -np.inf)
    bin_boundaries[-1] =  np.inf

    return bin_boundaries





