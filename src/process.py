"""
Module to define helper functions to load and process data.
"""

import pandas as pd

def load_data(file_path, source="csv"):
    """
    Load data from source.
    
    Parameters
    ----------
    file_path : str
        Path to the csv file.
    source : str
        Source of the data. Options:
            - csv (default): Load data from a local csv file.
                File path should be to a local csv file.
            - db: Load data from a database.
                File path should be to a query script.
    
    Returns
    -------
    pd.DataFrame
        Dataframe containing the data from the csv file.
    """
    if source == "csv":
        return pd.read_csv(file_path)
    # elif source == "db":
    #     return pd.read_sql(file_path)
    else:
        raise ValueError("Invalid source. Please select a valid source.")