import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculates the lower and upper thresholds for outlier detection based on quartile ranges.
    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame containing the column to compute outlier thresholds.
    - col_name (str): The name of the column in the DataFrame for which to calculate outlier thresholds.
    - q1 (float, optional): The percentile value for the first quartile. Defaults to 0.25.
    - q3 (float, optional): The percentile value for the third quartile. Defaults to 0.75.
    -***Note if data is skewed right or left use median value.***
    Returns:
    - tuple: A tuple containing the lower and upper thresholds for outliers. """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Checks if there are any outlier values in the specified column of a DataFrame based on quantile thresholds.

    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame to check for outliers.
    - col_name (str): The name of the column in which to search for outliers.
    - q1 (float, optional): The lower quantile to use for calculating the outlier detection threshold. Defaults to 0.25.
    - q3 (float, optional): The upper quantile to use for calculating the outlier detection threshold. Defaults to 0.75.

    Returns:
    - bool: True if outliers are found in the specified column; False otherwise. 
    """
    if pd.api.types.is_datetime64_any_dtype(dataframe[col_name]):
        # Handle datetime data differently or skip
        print(f"Column {col_name} is of datetime type, which is not suitable for outlier detection based on quartiles.")
        return False
    else:
        low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False


def grab_outliers(dataframe, col_name, index=False, head=5, q1=0.25, q3=0.75, print_table=True):
    """Prints the rows containing outliers in the specified column of the DataFrame, based on quantile thresholds.
    Optionally returns the indices of these outlier rows if requested.

    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame to analyze for outliers.
    - col_name (str): The name of the column to search for outliers.
    - index (bool, optional): Whether to return the indices of the rows containing outliers. Defaults to False.
    - head (int, optional): The number of outlier rows to print if there are more than 10. Defaults to 5.
    - q1 (float, optional): The lower quantile to calculate the outlier threshold. Defaults to 0.25.
    - q3 (float, optional): The upper quantile to calculate the outlier threshold. Defaults to 0.75.

    Returns:
    - pandas.Index (optional): If 'index' is True, returns the indices of rows containing outliers."""

    low, up = outlier_thresholds(dataframe, col_name,  q1=q1, q3=q3) #alt üst limtleri getiren foksiyon
    if print_table:
        if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10: # shape satır sayısı 10 dan büyük head ini alır
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(head))
        else:
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index



def remove_outlier(dataframe, col_name, q1=0.1, q3=0.99):
    """Removes outliers from a specified column in a pandas DataFrame based on the given quantile thresholds.
    
    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame from which outliers will be removed.
    - col_name (str): The name of the column in the DataFrame where outliers are to be identified and removed.
    - q1 (float, optional): The lower quantile to calculate the outlier threshold. Defaults to 0.1.
    - q3 (float, optional): The upper quantile to calculate the outlier threshold. Defaults to 0.99."""

    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers



def missing_values_table(dataframe, na_name=False):
    """
    Generates a table of columns with missing values, including the count of missing values and their proportion to the total number of entries.

    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame to analyze for missing values.
    - na_name (bool, optional): A flag to determine whether to return the names of columns with missing values. Defaults to False.

    Returns:
    - list (optional): If `na_name` is True, returns a list of column names that have missing values. Otherwise, no value is returned
    """

    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) #dataframe
    #ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) 
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False) 
    missing_df = pd.concat([n_miss, np.round(ratio, 3)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


