"""
This module helps you to ease your data science project. It contains seven functions those;
1. check_df()
    It gives general sight of dataframe objects.
2. cat_summary()
    It summarizes the categorical variables in the dataset
3. num_summary()
    It summarizes the categorical variables in the dataset
4. grab_col_names()
    It gives the names of categorical, numeric, and categorical but cardinal variables in the data set.
5. target_summary_with_cat()
    It gives the proportion and number of observation unit of the categorical variable about the target variable
6. target_summary_with_num()
    It gives the proportion of observation unit of the numerical variable about the target variable
7. high_correlated_cols
  It catches the high-correlated variables in your data set.


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
    print(f"##################### {numerical_col} #####################")
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




