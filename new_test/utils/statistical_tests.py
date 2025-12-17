import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from MFDFA import MFDFA
from diptest import diptest
from scipy.stats import skew, kurtosis
import scipy.stats as st


def moments(data, dataset = 'empirical'):
    '''
    Function to calculate the moments of a dataset.
    Parameters:
    data (pd.Series): Time series data to be analyzed.
    dataset (str): Name of the dataset. Default is 'empirical'.
    Returns: 
    tuple: Mean and standard deviation of the daily mean, standard deviation, skewness, and kurtosis of the dataset.
    '''
    mean_array = []
    std_array = []
    kurtosis_array = []
    skewness_array = []
    length_list_array = []
    # Group the data by date
    data_grouped = data.groupby(data.index.date)
    i = 0
    j = 0
    k = 0
    count = 0

    for segment in data_grouped:
        count += 1
        # choose only the segments with full data
        if (segment[1].isna().sum()) == 0:
            # check if the segment contains at least 20 hours of data
            if len(segment[1]) >  19*3600:
                i += 1
                length_list_array.append(len(segment[1]))
                mean_array.append(segment[1].mean())
                std_array.append(segment[1].std())
                kurtosis_array.append(kurtosis(segment[1], fisher = False))
                skewness_array.append(skew(segment[1]))
            else:
                j += 1
        else:
            k += 1

    mean_array = np.array(mean_array)
    std_array = np.array(std_array)
    kurtosis_array = np.array(kurtosis_array)
    skewness_array = np.array(skewness_array)

    print('Standardized moments of {} dataset:'.format(dataset))
    print('Daily mean of the {} mean: {:.3f} - Daily std of the mean: {:.3f}'.format(dataset, mean_array.mean(), mean_array.std()))
    print('Daily mean of the {} std: {:.3f} - Daily std of the std: {:.3f}'.format(dataset, std_array.mean(), std_array.std()))
    print('Daily mean of the {} skewness: {:.3f} - Daily std of the skewness: {:.3f}'.format(dataset, skewness_array.mean(), skewness_array.std()))
    print('Daily mean of the {} kurtosis: {:.3f} - Daily std of the kurtosis: {:.3f}'.format(dataset, kurtosis_array.mean(), kurtosis_array.std()))
    print(len(mean_array))
    print('Days with less than 20 hours:', j)
    print('Days with missing data:', k)
    print('Total number of days:', count)
    print('')
    return (mean_array.mean(), mean_array.std(), std_array.mean(), std_array.std(), skewness_array.mean(), skewness_array.std(), kurtosis_array.mean(), kurtosis_array.std())

def daily_profile(data,time_res = 1, tz = 'UTC', tz_to_convert = 'Asia/Seoul'): # !!! changed that to pandas operation!
    '''
    Function to calculate the daily profile of a time series.
    Parameters:
    data (pd.Series): Time series data to be analyzed.
    time_res (int): Time resolution for the daily profile. Default is 1.
    tz (str): Time zone of the input data. Default is 'UTC'.
    tz_to_convert (str): Time zone to convert the data to. Default is 'Asia/Seoul'.
    Returns:
    np.ndarray: Daily profile of the time series data.
    '''
    data.index = data.index.tz_localize(tz).tz_convert(tz_to_convert)
    day = data.groupby(data.index.time).mean()
    day = day[::time_res]
    
    return day.values


# Linearity test
def LTtest (data, stop = 30, step = 3, plot = False):
    '''
    Function to test the linearity of a time series using the CUSUM test.
    The CUSUM test is a statistical test that checks for changes in the mean of a time series.
    Parameters:
    data (pd.Series): Time series data to be tested.
    stop (int): Maximum number of lags to test. Default is 30.
    step (int): Step size for the lags. Default is 3.
    plot (bool): If True, plot the CUSUM test results. Default is False.'''

    # Compute Fourier transform 
    fft_0 = np.fft.fft(data)
    
    # Randomize phases of Fourier coefficients
    rand_phases = np.random.uniform(0, 2*np.pi, size=len(fft_0))
    surrogate = np.abs(fft_0) * np.exp(1j * rand_phases)
    
    # Compute inverse Fourier transform to obtain surrogate data
    surrogate = np.real(np.fft.ifft(surrogate))
    
    data = data[~np.isnan(data)]
    L = len(data)
    
    tau = np.arange(1, stop, step)
    #tau = np.arange(0, L//2, 3600)
    res_1 = np.zeros(len(tau))
    res_2 = np.zeros(len(tau))
    surr_1 = np.zeros(len(tau))
    surr_2 = np.zeros(len(tau))

    for i in range(len(tau)):
        x_t = data[0 :L-tau[i]]
        x_tau = data[tau[i]:L]
        y_t = surrogate[0:(L-tau[i])]
        y_tau = surrogate[(tau[i]):L]

        ## First method LT1
        # res_1[i] = np.mean(x_t**2 * x_tau)-np.mean(x_t * x_tau**2)
        # surr_1[i] = np.mean(y_t**2 * y_tau)-np.mean(y_t *y_tau**2)

        # Second method LT2
        res_2[i] = np.mean((x_t-x_tau)**3)/np.mean((x_t-x_tau)**2)
        surr_2[i] = np.mean((y_t-y_tau)**3)/np.mean((y_t-y_tau)**2)

    # Calculate the rmse(LT2)
    mse_lt2 = mean_squared_error(np.nan_to_num(res_2), np.nan_to_num(surr_2))
    rmse_lt2 = np.sqrt(mse_lt2)
    if plot == True:
        plt.figure(figsize=(10, 5))
        plt.plot(tau, res_2, label='Original Data')
        plt.plot(tau, surr_2, label='Surrogate Data')
        plt.xticks(np.arange(0, tau[-1], step))#labels=[f'{i // 3600}h' for i in np.arange(0, tau[-1], step)], fontsize=12)
        plt.legend()
    return rmse_lt2

def Linearity_test(data, stop = 30, step = 3, plot = False):
    '''Linearity test for full days
    Parameters:
    data (pd.Series): Time series data to be tested.
    stop (int): Maximum number of lags to test. Default is 30.
    step (int): Step size for the lags. Default is 3.
    plot (bool): If True, plot the CUSUM test results. Default is False.
    Returns:
    tuple: Mean and standard deviation of the RMSE values for each segment.
    '''
    np.random.seed(42)
    full_days_lt = []
    length_list = []
    data_grouped = data.groupby(data.index.date)
    for j,segment in enumerate(data_grouped):
        # choose only the days in the dataset as segments with at least 20 hours of data
        if len(segment[1]) > 19*3600:
            data = segment[1].dropna().values
            rmse_lt2 = LTtest(data, stop = 30, step = 3, plot=False)
            length_list.append(len(data))
            #print(rmse_lt2)
            full_days_lt.append(rmse_lt2)
    full_days_lt = np.array(full_days_lt)
    #print(full_days_lt)
    print(np.mean(full_days_lt), np.std(full_days_lt))
    return np.mean(full_days_lt), np.std(full_days_lt)

# Detrended Fluctuation Analysis (DFA) and Hurst exponent calculation
def Hurst_exponent_single_set(x,figsize = (8,6)):
    '''
    Function to calculate the Hurst exponent using Detrended Fluctuation Analysis (DFA).
    Parameters:
    x (pd.Series): Time series data to be analyzed.
    figsize (tuple): Size of the figure to be plotted. Default is (8, 6).
    Returns:
    Hurst (float): Estimated Hurst exponent.
    lag (np.ndarray): Array of lag values used in the analysis.
    dfa (np.ndarray): Array of DFA values corresponding to the lag values.
    '''

    # Define lag range for MFDFA analysis
    lag = np.logspace(0.7, 6, 100).astype(int)
    q = 2
    order = 1

    # Using valid_hours for the MFDFA analysis
    data = x.dropna().values
    lag, dfa = MFDFA(data, lag=lag, q=q, order=order)

    # Fit the first 10 points for the Hurst exponent estimation
    polyfit = np.polyfit(np.log(lag[:10]), np.log(dfa[:10]), 1)
    Hurst = polyfit[0] - 1


    '''Add text to display the slope '''
    # print(f"Slope for Valid Hours: {polyfit[0]}")

    return float(Hurst), lag, dfa
    
def Hurst_exponent(data, interval = 'd'):  
        '''
        Function to calculate the Hurst exponent for each segment of a dataset.
        Parameters:
        data (pd.Series): Time series data to be analyzed.
        interval (str): Time interval for grouping the data. Default is 'd' for daily.
        Returns:
        tuple: Mean and standard deviation of the Hurst exponent for each segment.
        '''
        # Group the data by the specified time interval
        Hurst_list = []
        for j,segment in enumerate(data.groupby(data.index.floor(interval))):
                # choose only days with at least 20 hours of data
                if len(segment[1]) > 19*3600:
                        H = Hurst_exponent_single_set(segment[1])[0]
                        Hurst_list.append(H)
        Hurst_list = np.array(Hurst_list)
        print(Hurst_list.mean(), Hurst_list.std())
        return Hurst_list.mean(), Hurst_list.std()


def dip_statistics(data, interval = 'd', alpha = 0.05):
    '''
    Function to calculate the DIP statistic for each segment of a dataset.
    Parameters:
    data: pd.Series: Frequency time series data.
    interval: str: Time interval for grouping the data (default is 'h' for hourly).
    Returns:
    tuple: DIP statistics and p-values for each segment. The p-values is defined as True (1) if the null hypothesis of unimodality is rejected.
    '''
    hourly_dip_statistics = np.array([])
    hourly_pval = np.array([])
    for _, segment in data.groupby(data.index.floor(interval)):
        # Calculate the DIP statistic for the 'freq' column
        dip_stat, pval = diptest(segment)
        pval = pval < alpha
        hourly_dip_statistics = np.append(hourly_dip_statistics, dip_stat) # Store the dip statistic for this hour
        hourly_pval = np.append(hourly_pval, pval) # Store the p-value for this hour
    return hourly_dip_statistics, pd.Series(hourly_pval)


def KL(data_empirical, data_synth, range = (59.85, 60.15)):
    '''
    Function to calculate the Kullback-Leibler divergence between two datasets.
    Parameters:
    data_empirical (pd.Series): Empirical data to be compared.
    data_synth (pd.Series): Synthetic data to be compared.
    range (tuple): Range for the histogram bins. Default is (59.85, 60.15).
    Returns:
    float: Kullback-Leibler divergence between the two datasets.
    '''
    # Calculate the histogram of the empirical and synthetic data
    hist_empirical,_ = np.histogram(data_empirical, bins=100, density=True,range = range)
    hist_syntht,_ = np.histogram(data_synth, bins=100, density=True,range = range)
    mask_empirical = hist_empirical == 0
    mask_synth = hist_syntht == 0
    KL_div = st.entropy(hist_empirical[~(mask_empirical + mask_synth)],hist_syntht[~(mask_empirical + mask_synth)])
    return KL_div


# still needs some work commenting!!!
def longest_true_streak(data_full_hours, boolean_array):
    dict_df = {}
    max_streak = 0
    current_streak = 0
    count = 0
    start_ind = 0 # = valid_hours.index[0]
    for i,value in enumerate(boolean_array[1:]):
        if value:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                end_streak = i
        else:
            current_streak = 0
            end_ind = i+1
            dict_df[count] = data_full_hours.iloc[start_ind:end_ind]
            start_ind = i+1
            count += 1 
    dict_df[count] = data_full_hours.iloc[start_ind:] # last component
    l = []
    for key in dict_df.keys():
        l.append(dict_df[key].shape[0]/3600)
    l = np.array(l)
    print('Number of consistent segments: ', len(l))
    longest_segment = l.argmax()
    return dict_df, start_ind, longest_segment # , max_streak,end_streak
