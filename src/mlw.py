# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:57:55 2022

@author: Tim G.
"""

# Below are functions provided by Discovery Corps, Inc.
## for use in the Machine Learning Workshop.
## They are provided as learning tools, without warranty, 
## either expressed or implied.

import pandas                     as     pd
import numpy                      as     np
import matplotlib.pyplot          as     plt
import seaborn                    as     sns              # another plotting package
import matplotlib.cm              as     cm
import csv
from sklearn.metrics              import roc_auc_score    # for measuring performance
from sklearn.metrics              import roc_curve        # for plotting performance
from sklearn.model_selection      import train_test_split # for partitioning a dataset
from sklearn.tree                 import DecisionTreeClassifier # for building a decision tree
from sklearn                      import tree                   # for the tree visualization
from statsmodels.stats.proportion import proportion_confint
from numpy                        import log as log
from scipy.stats                  import binom
from matplotlib.colors            import Normalize
from collections                  import OrderedDict


ROC_FIGURE_WIDTH        = 16
ROC_FIGURE_HEIGHT       = 16
ROC_ASPECT_RATIO        = 'equal'
ROC_CURVE_COLOR         = 'tab:blue'
ROC_DIAGONAL_COLOR      = 'tab:red'
ROC_LINE_STYLE          = '--'
ROC_LINE_WIDTH          = .75

TREE_PLOT_WIDTH      = 32
TREE_PLOT_HEIGHT     = 16
TREE_PLOT_LABEL      = "root"
TREE_PLOT_FILLED     = True
TREE_PLOT_ROUNDED    = True
TREE_PLOT_PROPORTION = True

EXTENSION_SEPARATOR_MAPPING = {'.tab':'\t',
                               '.csv':',',
                               '.ssv':';'
                               }



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
                       element_names=[],
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
    if len(element_names) > 0:
        working = pd.read_csv(data_folder_name + data_file_name, sep=file_separator,
                            header=None, names=element_names
                            )
    else:
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
                                            min_leaf=1000,
                                            tree_plot=False):
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
    tree_plot : Boolean, optional
        flag to turn on/off display of the tree plot.  The default is False.

    Returns
    -------
    (bin_boundaries, success) : dict, Boolean
        the dictionary of bin bounaries for the numeric elements
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
                                      tree_bin_min_leaf =min_leaf,
                                      display_tree_plot =tree_plot
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
                                 tree_bin_min_leaf=1000,
                                 display_tree_plot=False
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
    display_tree_plot : Boolean, optional
        flag to turn on/off display of the tree plot.  The default is False.

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
    
    # display the decision tree
    if display_tree_plot:
        plt.rcParams["figure.figsize"] = (TREE_PLOT_WIDTH, TREE_PLOT_HEIGHT)
        tree.plot_tree(tbc, feature_names=[predictor_series.name], 
                       label  =TREE_PLOT_LABEL, 
                       filled =TREE_PLOT_FILLED, 
                       rounded=TREE_PLOT_ROUNDED,
                       proportion=TREE_PLOT_PROPORTION)
        plt.title("tree binning for " + predictor_series.name, fontsize=36)
        plt.show()

    # assign a node id to each row of data
    node_id = pd.Series(tbc.apply(pd.DataFrame(predictor_series)))

    # extract the boundaries based on node_id
    bin_boundaries = list(predictor_series.groupby(node_id).max())

    # refine the first and last boundaries for data beyond the training range
    bin_boundaries.insert(0, -np.inf)
    bin_boundaries[-1] =  np.inf

    return bin_boundaries




def compute__rate_tables_for_all_category(predictors, target,
                                          target_rate,
                                          target_p_value=.0001,
                                          confidence_limits_p_value=.0001
                                          ):
    '''
    Parameters
    ----------
    predictors : dataframe
        the collection of predictors.
    target : series
        the target outcome.
    target_rate : float
        the fraction of positive outcomes.
    target_p_value : float, optional
        p-value for testing significance of a category rate versus the
        target rate. The default is .0001.
    confidence_limits_p_value : float, optional
        p-value for producing the confidence limits around the category rate.
        The default is .0001.

    Returns
    -------
    rate_tables : dict
        the dictionary of rate tables for the category elements
    '''
    # create empty dict for storing the rate tables
    rate_tables = {}

    # get the list of category elements
    category_element_names = list__category_elements(predictors)

    # if any, cycle thru all of the category elements and estimate rates
    for category_element_name in category_element_names:
        # use the function to compute the rate table
        rates = compute__rate_table(predictors[category_element_name], target, target_rate,
                                    target_rate_p_value=target_p_value,
                                    conf_limits_p_value=confidence_limits_p_value)

        # add the key-value pair for the category element name and the rates to a dict
        rates['element'] = category_element_name
        rates.index.name ='category'
        rate_tables[category_element_name] = rates

    return rate_tables




def list__category_elements(df):
    '''
    Parameters
    ----------
    df : pandas DataFrame
        this is usually the working DataFrame.

    Returns
    -------
    list
        the list of category elements that are present in the DataFrame.
    '''
    return df.select_dtypes(include='category').columns.tolist()




def compute__rate_table(predictor_series, target_series, target_rate,
                        target_rate_p_value=.0001,
                        conf_limits_p_value=.0001):
    '''
    Parameters
    ----------
    predictor_series : Series, numeric
        the predictor (input).
    target_series : Series, numeric, binary
        the target, or outcome of interest.
    target_rate : float
        rate (fraction) of positive outcomes for comparison to the predictor
        categories.
    target_rate_p_value : float, optional
        p-value for testing significance of a category rate versus the
        target rate. The default is .0001.
    conf_limits_p_value : float, optional
        p-value for producing the confidence limits around the category rate.
        The default is .0001.

    Returns
    -------
    rate_table : data frame
        table of relevant values associated with each category, including:
        the measured rate of positive outcomes, count of observations, 
        number of positive outcomes,
        flag (and probability) indicating significance versus the target rate,
        lower and upper confidence limits for the category rate,
        scoring rate to be used when applying to a category element
        (may differ) from the measured rate when the measured rate is NOT
        significantly different than the target rate.
    '''
    # start to compute the rate table
    base     = target_series.groupby(predictor_series)
    rate     = base.mean().fillna(target_rate)
    count    = base.count()
    positive = base.sum()
    
    rate_table         = pd.concat([rate, count, positive], axis=1)
    rate_table.columns = ['rate','count','positive']
    
    # test the category rate for significance versus the target rate
    rate_table['rate_test'] = rate_table.apply(
        lambda row : category_rate_test(
            row['count'], row['positive'], target_rate, sig_level=target_rate_p_value
            ),
        axis=1)
    rate_table[['flag','prob']] = pd.DataFrame(rate_table['rate_test'].tolist(), index=rate_table.index)
    rate_table.drop(columns=['rate_test'], inplace=True)
    
    # compute the lower and upper confidence limits around the rates
    rate_table['conf_lims'] = rate_table.apply(
        lambda row : compute__confidence_limits(row['count'], row['rate'], p_value=conf_limits_p_value),
        axis=1)
    rate_table[['rate_lower','rate_upper']] = pd.DataFrame(rate_table['conf_lims'].tolist(), index=rate_table.index)
    rate_table.drop(columns=['conf_lims'], inplace=True)
    
    # fill non-sig entries with the target rate for use when scoring
    rate_table['scoring_rate'] = rate_table['rate']
    rate_table.loc[rate_table['flag']==0, 'scoring_rate']=target_rate
    
    # compute the scoring_delta_rate as the difference from the target rate
    rate_table['scoring_delta_rate'] = rate_table['scoring_rate'] - target_rate

    return rate_table




def category_rate_test(ncat, npos, trate, sig_level=0.001, is_nmin=True):
    '''Test whether the category rate (of positives) is significantly different than
    the overall target rate for a dataset.
    
    Parameters:
        ncat is the number of records that fall in the category
        npos is the number of positive outcomes associated with the category
        trate is the target rate of positive outcomes for the dataset
        sig_level is the significance level of interest for the test
        is_nmin is a flag that a rule of thumb for the minimum number of records will be applied
        
    Returns:
        (flag, cat_prob) where:
            flag is -1, 1, or 0 if the category rate is deemed significantly 
                below, above, or not different from, respectively, the overall target rate
            cat_prob is the probability of observing the given category rate by chance
    '''
    
    # check if the category is too small to consider.  My criterion is that the category is big enough
    # to see at least one positive outcome with high probability (1 - sig_level)
    np.seterr(divide = 'ignore')
    if trate > 0 and trate < 1:
        nmin = log(sig_level)/log(1.0 - trate)  # see https://socratic.org/questions/how-do-you-find-the-probability-of-at-least-one-success-when-n-independent-berno
    elif trate == 1:
        nmin = 1
        print('100 percent rate of positive outcomes detected: review results carefully')
    else:
        nmin = np.inf
        print('zero percent rate of positive outcomes detected: review results carefully')
    np.seterr(divide = 'warn')

    if (ncat < nmin) and is_nmin:
        return 0, 0.5
    
    # compute the rate of positives for the category
    cat_rate = npos/ncat
    
    # mod the sig_level for the two-sided test
    sig_level = sig_level/2.0
    
    if cat_rate <= trate:    # use this branch if the category rate is below the target rate
        cat_prob = binom.cdf(k=npos,n=ncat,p=trate)
        if cat_prob <= sig_level:
            return -1, cat_prob
        else:
            return  0, cat_prob
    else:                    # use this branch if the category rate is at/above the target rate
        cat_prob = 1 - binom.cdf(k=npos,n=ncat,p=trate)
        if cat_prob <= sig_level:
            return  1, cat_prob
        else:
            return  0, cat_prob




def compute__confidence_limits(bin_count_avg, nom_target_rate, p_value=0.0001):
    '''
    Parameters
    ----------
    bin_count_avg : float
        expected or average number of observations in a bin.
    nom_target_rate : float
        the nominal target rate.
    p_value : float, optional
        the p-value for significance.  The default is .0001

    Returns
    -------
    (ci_low, ci_high) : tuple of floats
        the lower and upper confidence limits for a bin.
    '''

    positives_avg = nom_target_rate * bin_count_avg         # expected number of positive outcomes in an average bin
    if bin_count_avg > 0:
        (ci_low, ci_upp) = proportion_confint(positives_avg, bin_count_avg,
                                          alpha=p_value)    # to define the confidence interval
    else:
        (ci_low, ci_upp) = (0.0, 1.0)

    return (ci_low, ci_upp)




def apply__rate_tables_to_all_category(predictors, rate_tables, target_rate,
                                       suffix='_r'
                                       ):
    '''
    Parameters
    ----------
    predictors : dataframe
        the collection of predictors.
    rate_tables : dict
        the dictionary of rate tables for the category elements..
    target_rate : float
        the fraction of positive outcomes.
    suffix : string, optional
        the suffix used to form the names of the rate elements.  The default is '_r'.

    Returns
    -------
    predictors : dataframe
        the *updated* collection of predictors
    '''

    # if the rate_tables dictionary is not empty, find any available elements to rate
    if rate_tables:
        elements_with_rate_tables = list(rate_tables.keys())
        all_elements = predictors.columns.to_list()
        elements_to_rate = list(set(all_elements).intersection(elements_with_rate_tables))

        for element_name in elements_to_rate:
            # apply the (scoring) rates and add the new element to the dataset
            rate_element_name = element_name + suffix
            rate_element = apply__rate_table(predictors[element_name],
                                      rate_tables[element_name],
                                      target_rate)
            predictors = pd.concat([predictors, rate_element.rename(rate_element_name)], axis=1)

            # display rate counts
            #print(predictors[rate_element_name].value_counts())

    return predictors




def apply__rate_table(element, rate_table, nom_rate, use_delta_rate=True):
    '''
    Parameters
    ----------
    element : Series, category
        the categories to be translated into rates.
    rate_table : data frame
        table of relevant values related to the rate of positive outcomes.
    nom_rate : float
        nominal rate of positive outcomes, typically the target rate
    use_delta_rate : Boolean, optional
        flag to specify whether to use delta rates or base rates.  The default is True

    Returns
    -------
    rate_element : Series, numeric
        the rates associated with each category
    '''

    if use_delta_rate:
        rate_dict = rate_table['scoring_delta_rate'].to_dict()
        return element.map(rate_dict).astype('float').fillna(0.0)
    else:
        rate_dict = rate_table['scoring_rate'].to_dict()
        return element.map(rate_dict).astype('float').fillna(nom_rate)




def plot__what_matters_single_categories(rate_tables, target_rate, study_name,
                                         p_value=.0001, filename=""
                                         ):
    '''
    Parameters
    ----------
    rate_tables : dict
        the dictionary of rate tables for the category elements
    target_rate : float
        the fraction of positive outcomes.
    study_name : string
        name of the study, used for titling the plot.
    p_value : float, optional
        p-value for producing the confidence limits around the category rate.
        The default is .0001.
    filename : string, optional
        name of the file for saving the plot.  The default is "".

    Returns
    -------
    success : Boolean
        success flag.  Also produces the plot in the Viewer
    '''
    # define constants
    COUNT_SCALE_FACTOR     = 1.1        # 1.1 works fairly well to keep the bars from rubbing into each other
    MIN_BAR_SIZE           = 0.1        #  determined empirically based on the figure size
    PROFILE_COLOR_MAP      = 'RdYlBu_r' # 'RdBu_r' comes closest to my preferred Tableau colormap, 
                                        #  but I also like 'Spectral_r', 'RdYlBu_r', and others
                                        #  for more, see https://matplotlib.org/stable/tutorials/colors/colormaps.html
    TARGET_RATE_LINE_COLOR = 'tab:gray' # 'tab:gray' is visible but unobtrusive
    TARGET_RATE_BAND_COLOR = 'lightgray'
    TARGET_RATE_BAND_ALPHA = 0.3

    # capture the names of all of the elements represented in the rate_tables_dict
    element_names    = list(rate_tables.keys())
    is_first_element = True
    
    # across all elements, collect a roundup table of all of the significant categories
    mini_table_columns = ['element','rate','count','positive','flag']
    
    for element_name in element_names:
        mini_table  = rate_tables[element_name][mini_table_columns]
        mini_table  = mini_table.loc[mini_table['count']>0]
        mini_table  = mini_table.loc[mini_table['flag']!=0]
    
        if is_first_element:
            roundup = mini_table.copy()
            is_first_element=False
        else:
            roundup = pd.concat([roundup, mini_table])
    
    # sort the roundup table by rates, then by counts for ties
    roundup = roundup.sort_values(['rate','count'])
    
    # convert the category index to a string, then create the description
    roundup['category']    = roundup.index.astype(str)
    roundup['description'] = roundup['element'] + ' : ' + roundup['category']
    
    # make the descriptions a bit more readable
    roundup['description'] = roundup['description'].replace(
        {'\(-inf, ': '<',
         ', inf]'  : '>'
         }, regex=True)
    
    # determine appropriate sizes for the bars in the plot
    max_count = COUNT_SCALE_FACTOR * roundup['count'].max()     # adjustment to keep bars separated
    roundup['bar_size'] = roundup['count']/max_count
    condition = roundup['bar_size'] < MIN_BAR_SIZE
    roundup.loc[condition,'bar_size'] = MIN_BAR_SIZE
    
    # estimate confidence bands around the target rate using an average bin count
    bin_count_avg = roundup['count'].mean()
    (train_pct_ci_low, train_pct_ci_upp) = compute__confidence_limits(bin_count_avg, target_rate, p_value)
    
    # set the color range based on the deviation from the target rate
    if (roundup['rate'].min() < target_rate) and (roundup['rate'].max() > target_rate):
        color_delta = min(target_rate - roundup['rate'].min(), roundup['rate'].max() - target_rate)
        low_color   = target_rate - color_delta
        high_color  = target_rate + color_delta
    elif (roundup['rate'].min() > target_rate) and (roundup['rate'].max() > target_rate):
        color_delta = min(target_rate, roundup['rate'].max() - target_rate)
        low_color   = target_rate - color_delta
        high_color  = target_rate + color_delta
    else:
        color_delta = min(1.0 - target_rate, target_rate - roundup['rate'].min())
        low_color   = target_rate - color_delta
        high_color  = target_rate + color_delta

    my_norm     = Normalize(vmin=low_color, vmax=high_color)
    my_cmap     = cm.get_cmap(PROFILE_COLOR_MAP)
    
    # draw the plot, adjusting the figure size based on the number of rows in the roundup table
    figure_height = max(9, int(len(roundup)/5 + 1))
    plt.rcParams["figure.figsize"]  = (16, figure_height)
    plt.rcParams['lines.linestyle'] = '--'

    plt.barh('description', 'rate',data=roundup, height='bar_size',
             color=my_cmap(my_norm(roundup['rate'])))
    plt.axvspan(train_pct_ci_low, train_pct_ci_upp, alpha=TARGET_RATE_BAND_ALPHA,
                color=TARGET_RATE_BAND_COLOR)
    plt.axvline(target_rate, color=TARGET_RATE_LINE_COLOR)
    plt.title('What matters most for ' + study_name + ': ' + str(len(roundup)) + ' items')
    plt.xlabel("Rate")
    plt.ylabel("Element:Category")
    sns.despine()
    plt.savefig(filename)
    plt.show();

    return(True)




def export__dictionary(dict, filename, header):
    '''
    Parameters
    ----------
    dict : dictionary
        the dictionary object to export.
    filename : string
        the name to attach to the exported file.

    Returns
    -------
    success
        a success flag.
    '''
    ordered_dict = OrderedDict(sorted(dict.items()))  # to sort the dictionary by its keys
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(header)
        w.writerows(ordered_dict.items())

    return (True)



def combine__rate_tables(rate_tables):
    '''
    Parameters
    ----------
    rate_tables : dict
        the dictionary of rate tables to be combined into a single dataframe.

    Returns
    -------
    big_table : dataframe
        the combined table of rates.
    success : Boolean
        a success flag.
    '''
    # define constants, the desired order of the columns
    COLUMN_NAMES = ['element',
                    'category',
                    'rate',
                    'count',
                    'positive',
                    'flag',
                    'prob',
                    'rate_lower',
                    'rate_upper',
                    'scoring_rate',
                    'scoring_delta_rate'
                    ]

    # get the list of elements (there is one rate table for each)
    # and initialize the big_table
    element_names = list(rate_tables.keys())
    big_table = pd.DataFrame()

    # cycle thru all of the elements
    for element_name in element_names:
        mini_table  = rate_tables[element_name].copy()
        mini_table  = mini_table.loc[mini_table['count']>0]

        if (len(mini_table) > 0):
            mini_table['category'] = mini_table.index
            mini_table = mini_table.reset_index(drop=True)
            mini_table = mini_table.reindex(columns=COLUMN_NAMES)
            big_table = pd.concat([big_table, mini_table])

    big_table['category'] = big_table['category'].astype('category')
    big_table = big_table.sort_values(['element', 'category'])
    big_table = big_table.reset_index(drop=True)

    return (big_table, True)




def export__dataframe(df, filename, include_index=False):
    '''
    Parameters
    ----------
    df : dataframe
        the dataframe to export to a file.
    filename : string
        the name to attach to the exported file.
    include_index : Boolean, optional
        flag to include the index in the export. The default is False.

    Returns
    -------
    success : Boolean
        a success flag.
    '''
    # extract the column separator based on the file extension
    # only works for three-letter extensions
    extension = filename[-4:]
    separator = EXTENSION_SEPARATOR_MAPPING[extension]

    df.to_csv(filename, sep=separator, index=include_index)

    return True



