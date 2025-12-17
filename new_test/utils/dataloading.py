import pandas as pd

def load_data(data_path = "../../dataset/Frequency_data_SK.pkl", limit_interploation=10):
    '''
    Function to load and preprocess data.
    Parameters:
    ----
    data_path (str): Path to the data file. Default is "../../dataset/Frequency_data_SK.pkl".
    limit_interploation (int): Limit for interpolation of missing values. Default is 10.
    Returns:
    ----
    pd.DataFrame: Preprocessed DataFrame with missing values interpolated and quality indicators updated.
    '''
    
    data = pd.read_pickle(data_path)

    # select a limit to interpolate the missing values
    limit_interploation = 10

    ''' Add missing bad quality indicator'''
    data_0 = data.loc[data['QI']==0]
    ind = data_0[data_0['freq'].isna()].index[0]
    data.loc[ind,'QI'] = 2
    data.loc[:,'freq'] = data.loc[:,'freq'].interpolate(method='time',limit = limit_interploation)
    data.loc[data['freq'].isna(),'QI'] = 2
    data.loc[~data['freq'].isna(),'QI'] = 0
    return data

def load_data_full_hours(data):
    '''
    Function to load dataframe and filter it to keep only full hours.
    Parameters:
    data (pd.DataFrame): DataFrame to be filtered.
    Returns: 
    pd.DataFrame: Filtered DataFrame with only full hours.
    '''
    # Step 1: Load the data
    data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna()

    # Step 2: Group the data by the hour
    hourly_groups = data_filtered.groupby(data_filtered.index.floor('h'))

    # Step 3: Filter out incomplete hours 
    valid_hours = hourly_groups.filter(lambda x: len(x) == 3600)
    
    return valid_hours
