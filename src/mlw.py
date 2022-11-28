# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:57:55 2022

@author: Tim G.
"""

# Below are functions provided by Discovery Corps, Inc.
## for use in the Machine Learning Workshop.
## They are provided as learning tools, without warranty, 
## either expressed or implied.

import matplotlib.pyplot      as     plt
import seaborn                as     sns              # another plotting package
from sklearn.metrics          import roc_auc_score    # for measuring performance
from sklearn.metrics          import roc_curve        # for plotting performance
from sklearn.model_selection  import train_test_split # for partitioning a dataset

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
