# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:57:55 2022

@author: Tim G.
"""

# Below are functions provided by Discovery Corps, Inc.
## for use in the Machine Learning Workshop.
## They are provided as learning tools, without warranty, 
## either expressed or implied.

import matplotlib.pyplot as plt
import seaborn           as sns                  # another plotting package
from sklearn.metrics     import roc_auc_score    # for measuring performance
from sklearn.metrics     import roc_curve        # for plotting performance

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



