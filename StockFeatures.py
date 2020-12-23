
"""
Add stock features to the stock price history data to use in a machine learning algorithm. 
Examples: SMA, EMA, MACD, Bollinger percent

CD 12/2020  - Update to take in DataFrames instead of Numpy arrays. Add technical indicators as new columns to input data.
            - Add feature scaling
            - Remove output and initial data as a return variable. Just calculate features in this function.

"""

import numpy as np
import pandas as pd
import datetime
from Technical_Indicators import *

#Contains code to add features to the input data set when performing a machine learning algorithm

def add_features(price_minute_df,price_daily_df,ticker_output,features_list):
    """
    Add features to use for the machine learning algorithm. Returns the input data DateFrame.
    Sets the first column (index 0) of input data to the constant 1. Each new column of input data contains a new feature.
    CD 12/2020 - Update to use DataFrames, update feature labels in features_list, and add feature scaling

    price_minute_df - contains price data (open,close,high,low) and volume data for each ticker in minute level increments
                    - indexed by timestamps for each price point (index name - 'Datetime')
                    - a suffix of the ticker name is added to end of each column (Ex: open_GOOG, close_GOOG)
    price_daily_df - contains price data (open,close,high,low) and volume data for each ticker in day level increments
                    - indexed by timestamps for each price point (index name - 'Datetime')
                    - a suffix of the ticker name is added to end of each column (Ex: open_GOOG, close_GOOG)
    ticker_output - the stock ticker that is being predicted
    features_list - contains a list of strings each corresponding to a different feature to add to the input varaible
        if features_list = 'all', set features_list to the list of all the possible features

        Features 1 - 22 are calculated based on the output ticker #CD Change feature labels from numbers to strings
        ****1 - close price Should i include this?
        'close1min' - close price  1 minutes ago
        'close2min' - close price 2 minutes ago
        'close5min' - close price 5 minutes ago
        'close10min' - close price 10 minutes ago
        'close30min' - close price 30 minutes ago
        'close1hr' - close price 1 hour ago
        'DiffHighLow' - difference between high and low price
        ****9 - volume
        'SMA5' - 5 day simple moving average (SMA) 
        'SMA10' - 10 day simple moving average (SMA) 
        'SMA20' - 20 day simple moving average (SMA) 
        'SMA50' - 50 day simple moving avarerge (SMA)
        'EMA5' - 5 day exponential moving average (EMA) 
        'EMA10' - 10 day exponential moving average (EMA) 
        'EMA20' - 20 day exponential moving average (EMA) 
        'EMA50' - 50 day exponential moving avarerge (EMA) 
        'Boll20' - 20 day Bollinger Percent 
        'Boll50' - 50 day Bollinger percent 
        'MACDline' - MACD line 
        'SignalLine' - Signal line (9 day EMA of MACD line)
        'MACDhist' - MACD histogram
        'RSI' - Relative Strength Index (RSI)
        'Stochastic%K' - Stochastic Oscillator (%K)
        'Stochastic%K-%D' - Stochastic Oscillator (%K - %D)

    Returns a DataFrame of the input_data
        input_data - each row is a new training example, and each column is a new feature

    """

    #Temporarily add a 'Date' column that just contains the dates of price_minute_df and price_daily_df for calculations and merging
    price_minute_df['Date'] = price_minute_df['datetime'].dt.date
    price_daily_df['Date'] = price_daily_df['datetime'].dt.date

    input_data=price_minute_df.copy(deep=True)
    #Check if we are using all the features. If show, add all the features to the list.
    if features_list=='all':
        features_list = ['close1min','close2min','close5min','close10min','close30min','close1hr','DiffHighLow','SMA5','SMA10','SMA20',\
            'SMA50','EMA5','EMA10','EMA20','EMA50','Boll20','Boll50','MACDline','SignalLine','MACDhist','RSI','Stochastic%K','Stochastic%K-%D']

    """
    #Look as far back in the past as we need for the features list
    if 7 in features_list:
        past_time = 60  # 1 hour ago
    elif 6 in features_list:
        past_time = 30 # 30 minutes ago
    elif 5 in features_list:
        past_time = 10 # 10 minutes ago
    elif 4 in features_list:
        past_time = 5  # 5 minutes ago
    else:
        past_time = 2  # Default past_time = 2 minutes
    """

    #Initial data should be able to look into the past for more data and future for actual output (remove past_time and predict_time)
    #initial_data_dttm = dttm_data[past_time:len(dttm_data)-predict_time]
    #initial_data_price = data[ticker_output]['close'][past_time:len(dttm_data)-predict_time]

    #Output data is just initial data shifted by predict time
    #output_data = data[ticker_output]['close'][past_time+predict_time:len(dttm_data)]
    
    #Add a column of input data that contains only ones (bias unit)
    input_data['Bias'] = np.ones(input_data.shape[0])
    
    """ Use if we are not using day data
    #Find the indices in dttm_data for the close times of each day
    close_time_index = []
    local_date = datetime.datetime.fromtimestamp(dttm_data[0]/1000)
    for dttm_index in range(1,len(dttm_data)):
        local_date_prev = local_date
        local_date = datetime.datetime.fromtimestamp(dttm_data[dttm_index]/1000)
        if local_date.day != local_date_prev.day:
            #The previous dttm_index is the last value of the previous day
            close_time_index.append(dttm_index-1)
    """
    
    #if 1 in features_list:
        #Use the ticker_output price data
        #input_data = np.append(input_data,[initial_data_price],axis=0)
    
    if 'close1min' in features_list:
        #Use the close price of ticker output 1 minute ago
        input_data['close1min_' + ticker_output] = input_data['close_'+ticker_output].shift(periods=1)

    if 'close2min' in features_list:
        #Use the close price of ticker output 2 mintues ago
        input_data['close2min_' + ticker_output] = input_data['close_'+ticker_output].shift(periods=2)

    if 'close5min' in features_list:
        #Use the close price of ticker output 5 mintues ago
        input_data['close5min_' + ticker_output] = input_data['close_'+ticker_output].shift(periods=5)

    if 'close10min' in features_list:
        #Use the close price of ticker output 10 mintues ago
        input_data['close10min_' + ticker_output] = input_data['close_'+ticker_output].shift(periods=10)

    if 'close30min' in features_list:
        #Use the close price of ticker output 30 mintues ago
        input_data['close30min_' + ticker_output] = input_data['close_'+ticker_output].shift(periods=30)

    if 'close1hr' in features_list:
        #Use the close price of ticker output 1 hour ago
        input_data['close1hr_' + ticker_output] = input_data['close_'+ticker_output].shift(periods=60)

    if 'DiffHighLow' in features_list:
        #Use the difference between low and high price
        input_data['DiffHighLow_' + ticker_output] = input_data['high_'+ticker_output] - input_data['low_'+ticker_output]

    #if 9 in features_list:
        #Use the difference between low and high price
        #input_data = np.append(input_data,[data[ticker_output]['volume'][past_time:len(dttm_data)-predict_time]],axis=0)


    #Calculate the simple moving average for 5, 10, 20, and 50 days for the output ticker
    sma5 = calc_sma(price_minute_df,price_daily_df,'close_'+ticker_output,5)
    sma10 = calc_sma(price_minute_df,price_daily_df,'close_'+ticker_output,10)
    sma20 = calc_sma(price_minute_df,price_daily_df,'close_'+ticker_output,20)
    sma50 = calc_sma(price_minute_df,price_daily_df,'close_'+ticker_output,50)
    if 'SMA5' in features_list:
        #Use the 5 day simple moving average
        input_data['SMA5_' + ticker_output] = sma5

    if 'SMA10' in features_list:
        #Use the 10 day simple moving average
        input_data['SMA10_' + ticker_output] = sma10

    if 'SMA20' in features_list:
        #Use the 20 day simple moving average
        input_data['SMA20_' + ticker_output] = sma20
    
    if 'SMA50' in features_list:
        #Use the 50 day simple moving average
        input_data['SMA50_' + ticker_output] = sma50

    
    #Calculate the exponential moving average for 5, 10, 20, and 50 days for the output ticker
    (ema5,_) = calc_ema(price_minute_df,price_daily_df,'close_'+ticker_output,5)
    (ema10,_) = calc_ema(price_minute_df,price_daily_df,'close_'+ticker_output,10)
    (ema20,_) = calc_ema(price_minute_df,price_daily_df,'close_'+ticker_output,20)
    (ema50,_) = calc_ema(price_minute_df,price_daily_df,'close_'+ticker_output,50)
    if 'EMA5' in features_list:
        #Use the 5 day exponential moving average
        input_data['EMA5_' + ticker_output] = ema5

    if 'EMA10' in features_list:
        #Use the 10 day exponential moving average
        input_data['EMA10_' + ticker_output] = ema10

    if 'EMA20' in features_list:
        #Use the 20 day exponential moving average
        input_data['EMA20_' + ticker_output] = ema20
    
    if 'EMA50' in features_list:
        #Use the 50 day exponential moving average
        input_data['EMA50_' + ticker_output] = ema50

    
    #Calculate the 20 day and 50 day standard deviation of the output ticker
    std20 = calc_std(price_minute_df,price_daily_df,'close_'+ticker_output,sma20,20)
    std50 = calc_std(price_minute_df,price_daily_df,'close_'+ticker_output,sma50,50)
    if 'Boll20' in features_list:
        #Use the 20 day Bollinger Percent, Note: we are using 2*standard deviation to get the lower and upper band
        boll_lower_20 = sma20-2*std20
        boll_upper_20 = sma20+2*std20
        boll_percent_20 = (price_minute_df['close_'+ticker_output]-boll_lower_20)/(boll_upper_20-boll_lower_20)*100
        input_data['Boll20_' + ticker_output] = boll_percent_20

    if 'Boll50' in features_list:
        #Use the 50 day Bollinger Percent, Note: we are using 2.1*standard deviation to get the lower and upper band
        boll_lower_50 = sma50-2.1*std50
        boll_upper_50 = sma50+2.1*std50
        boll_percent_50 = (price_minute_df['close_'+ticker_output]-boll_lower_50)/(boll_upper_50-boll_lower_50)*100
        input_data['Boll50_' + ticker_output] = boll_percent_50


    #Calculate the MACD (Moving Average Convergence/Divergence Oscillator)
    [macd_line,signal_line,macd_hist] = calc_macd(price_minute_df,price_daily_df,'close_'+ticker_output)
    if 'MACDline' in features_list:
        #Use the MACD line
        input_data['MACDline_' + ticker_output] = macd_line

    if 'SignalLine' in features_list:
        #Use the Signal line (9 day EMA of MACD line)
        input_data['SignalLine_' + ticker_output] = signal_line

    if 'MACDhist' in features_list:
        #Use the MACD histogram (MACD_line - Signal line)
        input_data['MACDhist_' + ticker_output] = macd_hist

    if 'RSI' in features_list:
        #Use the Relative Strength Index
        (rsi,_) = calc_rsi(price_minute_df,price_daily_df,'close_'+ticker_output)
        input_data['RSI_' + ticker_output] = rsi


    #Get the 14 day stochastic oscillator
    (stochastic_k_line,stochastic_d_line,_s) = calc_stochastic_oscillator(price_minute_df,price_daily_df,ticker_output,14)
    if 'Stochastic%K' in features_list:
        #Use the %K of the Stochastic Oscillator
        input_data['Stochastic%K_' + ticker_output] = stochastic_k_line

    if 'Stochastic%K-%D' in features_list:
        #Use the difference between the %K and %D of the Stochastic Oscillator
        input_data['Stochastic%K-%D_' + ticker_output] = stochastic_k_line-stochastic_d_line
    

    #Remove the 'Date' column in price_minute_df and price_daily_df that was only used for calculations
    price_minute_df = price_minute_df.drop(columns=['Date'])
    price_daily_df = price_daily_df.drop(columns=['Date'])

    #CD 12/2020 Add feature scaling, use min-max scaling to keep values between 0 and 1\
    drop_columns = []
    for column in input_data.columns:
        if (column != 'datetime') & (column != 'Date') & (column != 'Bias'):
            input_data[column] = (input_data[column]-input_data[column].min())/(input_data[column].max()-input_data[column].min())

            #CD 12/2020 Remove features that contain only NaN rows. Ex: volume data for SPX, NDX, or DJX
            if input_data[column].isnull().all():
                drop_columns.append(column)

    input_data.drop(columns=drop_columns,inplace=True)

    return input_data
    



"""
#Get the close data for a particular date
test_df = input_data[['datetime','close_'+ticker_output,'SMA5_'+ticker_output,'SMA20_'+ticker_output,'SMA50_'+ticker_output]]
test = test_df[(test_df['datetime'].dt.hour==17) & (test_df['datetime'].dt.minute==0)]
test

test_df = input_data[['datetime','close_'+ticker_output,'Boll20_'+ticker_output,'Boll50_'+ticker_output]]
test = test_df[(test_df['datetime'].dt.hour==17) & (test_df['datetime'].dt.minute==0)]
test

test2_df = input_data[['datetime','close_'+ticker_output,'Boll20_'+ticker_output,'Boll50_'+ticker_output]]
test2 = test2_df[(test2_df['datetime'].dt.month==11) & (test2_df['datetime'].dt.day==30)]
test2

test.plot(kind='line',x='datetime',y='Boll20_'+ticker_output)
ax = plt.gca()
test.plot(kind='line',x='datetime',y='Boll50_'+ticker_output,ax=ax)
plt.show()

input_data.plot(kind='line',x='datetime',y='Stochastic%K_'+ticker_output)
ax = plt.gca()
input_data.plot(kind='line',x='datetime',y='Stochastic%D_'+ticker_output,ax=ax)

#Plot verticle lines at the regular hours market close
close_times = input_data[(input_data['datetime'].dt.hour==13) & (input_data['datetime'].dt.minute==0)]['datetime']
for close in close_times:
    plt.axvline(x=close)
plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
fig, ax1 = plt.subplots(1, 1)
x_data = test2['datetime']
x = np.arange(len(x_data))
ax1.plot(x,test2['RSI_'+ticker_output],label="RSI")
N = len(x_data)
fmt="%Y-%m-%d"
def format_date(index, pos):
    index = np.clip(int(index + 0.5), 0, N - 1)
    return x_data[index].strftime(fmt)
ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
fig.autofmt_xdate()
ax1.legend()
plt.show()

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
fig, ax1 = plt.subplots(1, 1)
x_data = input_data['datetime']
x = np.arange(len(x_data))
ax1.plot(x,input_data['close_'+ticker_output],label="Price")
ax1.plot(x,boll_lower_50,label="Lower Boll")
ax1.plot(x,boll_upper_50,label="Upper Boll")
ax1.plot(x,sma50,label="SMA")
N = len(x_data)
fmt="%Y-%m-%d"
def format_date(index, pos):
    index = np.clip(int(index + 0.5), 0, N - 1)
    return x_data[index].strftime(fmt)
ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
fig.autofmt_xdate()
ax1.legend()
plt.show()




#More simple way to display results
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
fig, ax1 = plt.subplots(1, 1)
x_data = initial_data_dttm
x = np.arange(len(x_data))
ax1.plot(x,initial_data_price,label="Price")
ax1.plot(x,boll_lower_50,label="Lower Boll")
ax1.plot(x,boll_upper_50,label="Upper Boll")
ax1.plot(x,sma50,label="SMA")
ax1.legend()
plt.show()



import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
fig, ax1 = plt.subplots(1, 1)
x_data = initial_data_dttm
x = np.arange(len(x_data))
ax1.plot(x,initial_data_price,label="Price")
ax1.plot(x,ema5,label="EMA 5")
ax1.plot(x,ema10,label="EMA 10")
ax1.plot(x,ema20,label="EMA 20")
ax1.plot(x,ema50,label="EMA 50")
N = len(x_data)
fmt="%Y-%m-%d"
def format_date(index, pos):
    index = np.clip(int(index + 0.5), 0, N - 1)
    return x_data[index].strftime(fmt)
ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
fig.autofmt_xdate()
ax1.legend()
plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
fig, ax1 = plt.subplots(1, 1)
x_data = initial_data_dttm
x = np.arange(len(x_data))
ax1.plot(x,rsi,label="RSI")
ax1.plot(x,stochastic_k_line,label="%K")
ax1.plot(x,stochastic_d_line,label="%D")
N = len(x_data)
fmt="%Y-%m-%d"
def format_date(index, pos):
    index = np.clip(int(index + 0.5), 0, N - 1)
    return x_data[index].strftime(fmt)
ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
fig.autofmt_xdate()
ax1.legend()
plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
fig, ax1 = plt.subplots(1, 1)
ax1.plot(x,boll_percent_20,label="Boll 20")
ax1.plot(x,boll_percent_50,label="Boll 50")
ax1.legend()
plt.show()

"""