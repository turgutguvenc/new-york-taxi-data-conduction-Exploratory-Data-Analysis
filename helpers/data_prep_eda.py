"""
This module helps you to ease your data science project. It contains seven functions those;

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def check_df(dataframe, head=5, tail=5, detail=False):
    """
    It gives general sight of dataframe objects.
    Parameters
    ----------
    dataframe: dataframe
        dataframe from which variable(column) names are to be retrieved.
    head: int, default 5
        It determines that how many of the first rows will print.
    tail: int,  default 5
        It determines that how many of the last rows will print.
    detail: boolean, default False
        It gives quantiles values
    Returns
    -------
        this is function don't return anything.It just prints summarized values
    Examples
    ------
    import seaborn as sns
    df = sns.load_dataset("tips")
    print(check_df(df,detail=True))
    """

    print("##################### Index #####################")
    print(dataframe.index)
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### Duplicated #####################")
    print(dataframe.duplicated().any())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Total NA #####################")
    print(dataframe.isnull().sum().sum())
    if detail:
        #print("##################### Quantiles #####################")
        #print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
        print("##################### Describe #####################")
        print(dataframe.describe(include = 'all').T)




#######################################################

def cat_summary(dataframe, col_name, plot=False, figsize=(5, 3)):
    """
    It summarizes the categorical variables in the dataset
    Parameters
    ----------
    dataframe: dataframe
        dataframe from which variable(column) names are to be retrieved.
    col_name: string
        categorical column name in dataframe
    plot : boolean
        This is the optional selection to make a boxplot graph.

    Examples
    ------
    import seaborn as sns
    df = sns.load_dataset("tips")
    print(cat_summary(df, "sex", plot=True))
    """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        plt.figure(figsize=figsize)
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=90)
        plt.show()




def num_summary(dataframe, numerical_col, plot=False, plot_type="hist", figsize=(5, 3)):
    """
     It summarizes the categorical variables in the dataset
    Parameters
        ----------
        dataframe: dataframe
            dataframe from which variable(column) names are to be retrieved.
        numerical_col: string
        plot: boolean (default=False)
            It makes default histogram graph.
        plot_type : string ( (default=hist) , hist or boxplot)
            It makes defaultly histogram graph you can change it to "boxplot" to make for box plot graph
    Examples
    ------
    import seaborn as sns
    df = sns.load_dataset("tips")
    print(num_summary(df, "tip", plot=True))
    """

    print("##################### Describe #####################")
    print(dataframe[numerical_col].describe(), "\n\n")
    print("##################### Total NA #####################")
    print(dataframe.isnull().sum().sum())
    if plot:
        plt.figure(figsize=figsize)
        if plot_type == "hist":
            dataframe[numerical_col].hist(bins=30)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()

        elif plot_type == "box_plot":
            sns.boxplot(x=dataframe[numerical_col])
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()
        else:
            print("Please enter the correct graph name!!!")



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numeric, and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ----------
    dataframe: dataframe
        dataframe where variable names want to be imported
    cat_th: int, optional
        The threshold for those categorical variables with the numerical appearance
    car_th: int, optional
        The threshold those categorical but cardinal variables

    Returns
    -------
        cat_cols: list
            List of categorical variables
        num_cols: list
            list of numerical variables
        cat_but_car: list
            list of categorical but cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total variables
       The cat_cols contains the num_but_cat in

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car



def high_correlated_cols(dataframe, plot=False, corr_th=0.90, fig_id=None, IMAGES_PATH=None):
    """
    It catches the high-correlated variables in your data set.
    Parameters
    ----------
    dataframe: dataframe
        dataframe where variable names want to be imported
    plot: boolean (default False)
        This is the optional selection to make a heatmap graph
    corr_th: float optional
        This argument determines the correlation threshold
    Returns
    -------
    drop_list: list
        List of high correlative variables. (default=0.90)
    Examples
    ----------
    import seaborn as sns
    df = sns.load_dataset("breast_cancer")
    high_correlated_cols(df, plot=False, corr_th=0.90)
    """

    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.title("Correlation Heat Map")
        if fig_id:  # Check if fig_id is provided
            path = IMAGES_PATH / f"{fig_id}.png"
                
            plt.tight_layout()
            plt.savefig(path, format="png", dpi=300)
        plt.show()
    return drop_list



def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculates the lower and upper thresholds for outlier detection based on quartile ranges.
    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame containing the column to compute outlier thresholds.
    - col_name (str): The name of the column in the DataFrame for which to calculate outlier thresholds.
    - q1 (float, optional): The percentile value for the first quartile. Defaults to 0.25.
    - q3 (float, optional): The percentile value for the third quartile. Defaults to 0.75.

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


