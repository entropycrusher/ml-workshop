# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:23:45 2022

@author: tgrae
"""

def compute__macro_stats(df, 
                         benchmark_folder_name='../bmrk/', 
                         study_name='bank-telemarketing', 
                         benchmark_separator='\t'):
    '''
    Parameters
    ----------
    df : dataframe
        the working dataframe for which you want to compute the stats.
    benchmark_folder_name : string, optional
        the path to the benchmark folder for the output. The default is '../bmrk/'.
    study_name : string, optional
        the name of the study to use in naming the output. The default is 'bank-telemarketing'.
    benchmark_separator : string, optional
        the separator to use for the output files. The default is '\t'.

    Returns
    -------
    stats_numeric : dataframe
        the table of stats for the numeric elements
    stats_object  : dataframe
        the table of stats for the object  elements
        
    None.  Produces two output files, one for numeric elements, one for object elements

    '''
    stats_numeric = df.describe(include=['number']).T
    stats_object  = df.describe(include=['object']).T
    
    stats_numeric_filename = benchmark_folder_name + study_name + "_numeric_stats.tab"
    stats_object_filename  = benchmark_folder_name + study_name + "_object_stats.tab"

    stats_numeric.to_csv(stats_numeric_filename, sep=benchmark_separator, index=True)
    stats_object.to_csv( stats_object_filename,  sep=benchmark_separator, index=True)

    return(stats_numeric, stats_object)

### Use your new function
test_num, test_obj = compute__macro_stats(working)
