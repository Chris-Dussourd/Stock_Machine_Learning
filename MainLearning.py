"""
Main file to be called for performing Machine Learning on stock price data (see main start function at bottom)
    1. Loads stock price data from pickle files using folder structure in file_organization.py
    2. Aligns data between tickers and creates minute level and day level pandas DataFrames
    3. Pass the input data into a learning algorithm (Linear Regression and Neural Networks are the only ones supported so far)
    4. Evaluate the learning algorithm by predicting profit, calculating performance measures (e.g. accuracy), and plotting data.

This was created from the first version posted of Linear Regression (9/2020). 
The old version loads the stock price data into numpy arrays and linear regression to guess the output price.

CD 12/2020  - Remove code for linear regression and update code to load stock price data into pandas Dataframe
            - Support a period of time between start_date and end_date and create a datetime list of federal holidays
            - Load dictionary of folder structure so we can automatically get ticker data
            - Add in new Neural Network algorithm and functions to evaluate learning algorithms

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime,math,pickle,os,calendar
from dateutil import tz
from StockFeatures import add_features
from pathlib import Path
import file_organization as foldstruct
from LinearRegression import linear_reg_normal
from NeuralNetworks import neuralnet_category_sigmoid


def stock_machine_learning(price_minute_df,price_daily_df,ticker_output,predict_period,predict_periodType,features_list,algorithm):
    """
    Get features for the stock price data and perform machine learning on stock price data and evaulate the algorithms
    CD 12/2020 Make into a function

    price_minute_df - DataFrame of minute level data with datetime values as the index
                      ['open_GOOG','close_GOOG','low_GOOG','high_GOOG','volume_GOOG'] - example columns for ticker GOOG (Google)
    price_daily_df - DataFrame of day level data with datetime values as the index 
                     Include data 6 months before price_df to allow calculation of technical indicators such as SMA
    ticker_output - the stock ticker that is being predicted
    predict_period - the amount of time in the future the algorithm is predicting
    predict_periodType - unit of time to predict in future (only supports 'minute' and 'day')
    features_list - list of features of stock data used in input data
    algorithm - the machine learning algorithm to use (only 'Linear Regression' for now)
    """ 
    lamb = 1000 #Regularization parameter used to prevent overuse of a few features
    input_data_df = add_features(price_minute_df,price_daily_df,ticker_output,features_list)

    #Create the output_data_df that the algorithm is trying to predict
    if predict_periodType == 'minute':
        #Shift the close price data for the output ticker back by predict_period minutes
        output_data_df = input_data_df['close_'+ticker_output].shift(periods=-predict_period).reset_index(drop=True)
    elif predict_periodType == 'day':
        #Find number of minute data points are in one day (Should be 781 - 13 hours of regular+extended hours trading + final minute)
        minutes_inday = len(input_data_df[input_data_df['Date']==input_data_df.loc[0,'Date']]) 
        #Shift the close price data for the output ticker back by predict_period days 
        output_data_df = input_data_df['close_'+ticker_output].shift(periods=-(predict_period*minutes_inday)).reset_index(drop=True)
    else:
        raise ValueError("This algorithm only support predict period types of 'day' and 'minute.'")

    #Remove 'Date' and 'datetime' column in input_data (we are not using the date as a feature)
    input_data_df = input_data_df.drop(columns=['datetime','Date'])

    #Find the rows in the input and ouptut data that are NaN (due to shifting of data)
    input_nan = input_data_df.isnull().any(axis=1)
    output_nan = output_data_df.isnull()
    nan_rows = ((input_nan) | (output_nan))

    #Remove the nan rows from input and output data
    input_data_df = input_data_df[~nan_rows]
    output_data_df = output_data_df[~nan_rows]
    
    #Separate input and output data into training (2/3 of data) and test sets (1/3 of data)
    train_input_df = input_data_df.loc[0:math.floor(input_data_df.shape[0]*2/3)]
    train_output_df = output_data_df.loc[0:math.floor(output_data_df.shape[0]*2/3)]
    test_input_df = input_data_df.loc[math.floor(input_data_df.shape[0]*2/3+1):]
    test_output_df = output_data_df.loc[math.floor(input_data_df.shape[0]*2/3+1):]

    #Run the machine learning algorithm on the data
    if algorithm == 'Linear Regression':
        (theta,train_predicted_df) = linear_reg_normal(train_input_df,train_output_df,lamb)
        test_predicted_df = np.dot(test_input_df,theta)

    elif algorithm == 'Neural Network':
        #Divide the output data into categories depending on percent loss/gain from input
        threshold = [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1]
        train_output_cat_df = train_output_df.copy(deep=True)
        test_output_cat_df = test_output_df.copy(deep=True)
        percent_gain_df = (output_data_df-input_data_df['close_'+ticker_output])/input_data_df['close_'+ticker_output]

        for i in range(len(threshold)+1):
            #Update output into categories for each threshold value
            if i == 0:
                train_output_cat_df[percent_gain_df<=threshold[i]] = i
                test_output_cat_df[percent_gain_df<=threshold[i]] = i
            elif i == len(threshold):
                train_output_cat_df[percent_gain_df>threshold[i-1]] = i
                test_output_cat_df[percent_gain_df>threshold[i-1]] = i
            else:
                train_output_cat_df[(percent_gain_df<=threshold[i]) & (percent_gain_df>threshold[i-1])] = i
                test_output_cat_df[(percent_gain_df<=threshold[i]) & (percent_gain_df>threshold[i-1])] = i

        #Call the neural network training algorithm with a sigmoid activation function
        hidden_layers=2
        hidden_units=100
        model = neuralnet_category_sigmoid(train_input_df,train_output_cat_df,hidden_layers,hidden_units)
        
        #Evaluate the test set data
        model.evaluate(x=test_input_df,y=test_output_cat_df)

        #Predict the values for the train and test sets
        train_predicted_df = pd.DataFrame(model.predict(train_input_df))
        test_predicted_df = pd.DataFrame(model.predict(test_input_df))
        train_predicted_cat_df = train_predicted_df.idxmax(axis=1)
        test_predicted_cat_df = test_predicted_df.idxmax(axis=1)

    test=1



def extract_data(tickers_input,start_date,end_date):
    """
    CD 12/2020 Make into a function
    Extract out the price and volume ticker data from pickle data (folder locations in file_structure_dict) and return as a DataFrame

    tickers_input - the stock tickers used as input to the machine learning algorithm
    start_date - the start time period to extract minute level input ticker data from
    end_date - the end time period to extract minute level input ticker data to

    Return price_df and price_daily_df
        price_df - DataFrame of minute level data from start_date to end_date
            Index - datetime values
            Ex columns: ['open_GOOG','close_GOOG','low_GOOG','high_GOOG','volume_GOOG'] - example columns for ticker GOOG (Google)
        price_daily_df - DataFrame of day level data from 6 months before start_date to end_date 
            Return 6 months before start_date to calculate technical indicators such as SMA
    """

    #Get the directory of where the folders of stock price data are located.
    directory = foldstruct.folder_directory

    #Get daily data 6 months before minute level data. Used to calculate technical indicators such as SMA
    start_daily = start_date.replace(month=start_date.month-6) if start_date.month>6 else start_date.replace(year=start_date.year-1,month=start_date.month+6)

    #CD 12/2020 Dictionary of dataframes (each key holds a dataframe of stock prices/datetimes/volume for one ticker)
    price_dict_df = {} 
    price_daily_dict_df = {}

    #Find the pickle files that will be used as input data and save them into a temp_history dictionary. CD 12/2020 support DataFrames
    for ticker in tickers_input:
        
        #Temporary dataframe that holds data for only one ticker
        price_df_temp = pd.DataFrame()
        price_daily_df_temp = pd.DataFrame()

        #Get the folder where this ticker is located
        folder = foldstruct.folder_structure[ticker]
        path = os.path.join(directory,folder)

        #CD 12/2020 Find files that correspond to the date we want to obtain using file_structure to get the start_date/end_date
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path,f)):
                
                #Remove the .p from the filename
                f = f[:-2]

                #Get start date and end date from filename and convert to date
                startDate = f.split('_')[foldstruct.file_structure.index('vStartDate')]
                startDate = datetime.date(int(startDate[0:4]),int(startDate[4:6]),int(startDate[6:]))
                endDate = f.split('_')[foldstruct.file_structure.index('vEndDate')]
                endDate = datetime.date(int(endDate[0:4]),int(endDate[4:6]),int(endDate[6:]))

                #Find files that overlap with the time range and store them in a DataFrame
                if f.split('_')[foldstruct.file_structure.index('vFrequency')]=='Minute' and startDate<=end_date and endDate>=start_date:
                    temp_history=pickle.load(open(os.path.join(path,f+'.p'),"rb"))
                    price_df_temp = price_df_temp.append(pd.DataFrame(temp_history[ticker]),ignore_index=True)
                #Find files that overlap the daily time range and store them in the dialy DataFrame
                elif f.split('_')[foldstruct.file_structure.index('vFrequency')]=='Daily' and startDate<=end_date and endDate>=start_daily:
                    temp_history=pickle.load(open(os.path.join(path,f+'.p'),"rb"))
                    price_daily_df_temp = price_daily_df_temp.append(pd.DataFrame(temp_history[ticker]),ignore_index=True)

        #Sort dataframes by datetime
        price_df_temp.sort_values(by='datetime',inplace=True)
        price_daily_df_temp.sort_values(by='datetime',inplace=True)

        #Remove duplicates (in case we have files with overlapping data)
        price_df_temp.drop_duplicates(inplace=True)
        price_daily_df_temp.drop_duplicates(inplace=True)

        #Change the datetime timestamps of int64 type to datetime objects using Pacific time zone
        price_df_temp['datetime'] = pd.to_datetime(price_df_temp['datetime'],unit='ms').dt.tz_localize('utc').dt.tz_convert(tz.gettz('US/Pacific'))
        price_daily_df_temp['datetime'] = pd.to_datetime(price_daily_df_temp['datetime'],unit='ms').dt.tz_localize('utc').dt.tz_convert(tz.gettz('US/Pacific'))

        #Save the dataframe into a dictionary
        price_dict_df[ticker] = price_df_temp
        price_daily_dict_df[ticker] = price_daily_df_temp

    #Align all the ticker datetimes and merge into one DataFrame
    price_minute_df = align_data(price_dict_df,start_date,end_date,'minute')
    price_daily_df = align_data(price_daily_dict_df,start_daily,end_date,'daily')

    #Index price_minute_df by date and time (multi-index) and price_daily_df by date
    #price_minute_df.index = pd.MultiIndex.from_arrays([price_minute_df.index.date,price_minute_df.index.time],names=['Date','Time'])
    #price_daily_df.index = pd.Index(price_daily_df.index.date,name='Date')

    return (price_minute_df,price_daily_df)



def align_data(price_dict_df,start_date,end_date,frequencyType):
    """
    Many times the saved prices recorded by TD Ameritrade are recorded at slightly different times.
    Ex: Getting the minute level prices of MSFT and AAPL between June 1st and June 2nd will not return the same number of prices nor the same price timings.
        Some minutes are skipped and therefore it throws off the indexes

    This function approximately aligns the times/prices in price_history to datetimes between start_date and end_date, so they are the same length for future computations.
    CD 12/2020 Support DataFrames and add support to ignore holidays

    price_dict_df - dictionary of DataFrames (key=ticker, values=DataFrame with columns ['datetime','open','close','high','low','volume'])
    start_date - date that represents when the data should start
    end_date - date that represents when the data should end
    frequencyType - 'minute' or 'daily' for how frequently we need data stored in the DataFrames

    Returns price_df - a dataframe of stock price info
        - first column level contains the ticker
        - second column level contains open, close, high, low, and volume
        - indexed by datetimes
    """
    #Get the holiday dates within date range
    holidays = get_holidays(start_date,end_date)

    datetime_list = []
    #Datetime for every day between time range
    if frequencyType=='daily':
        #The daily price data is saved at 10pm Pacific Time on TD Ameritrade site. Use this time for easier merge.
        local_datetime = datetime.datetime.combine(start_date,datetime.time(hour=22),tzinfo=tz.gettz('US/Pacific'))
        while local_datetime <= datetime.datetime.combine(end_date,datetime.time(hour=22),tzinfo=tz.gettz('US/Pacific')):

            #Skip weekends (Saturday and Sunday)
            if local_datetime.weekday() == 5:
                local_datetime=local_datetime+datetime.timedelta(days=2)
            #Skip Sundays and holidays
            elif (local_datetime.weekday() == 6) or (local_datetime.date() in holidays):
                local_datetime=local_datetime+datetime.timedelta(days=1)
            else:
                #Save datetime sinto list and increase local date by one day
                datetime_list.append(local_datetime)
                local_datetime=local_datetime+datetime.timedelta(days=1)
    
    #Datetime for every minute between time range
    elif frequencyType=='minute':
        #Start at 4am on the same day as start_date (Extended hours trading begins at 4am Pacific time on TD Ameritrade) 
        local_datetime = datetime.datetime.combine(start_date,datetime.time(4),tzinfo=tz.gettz('US/Pacific'))
        while local_datetime < datetime.datetime.combine(end_date,datetime.time(17,1),tzinfo=tz.gettz('US/Pacific')):

            #Extended hours ends at 5pm Pacific time, so skip to 4am the next day
            if local_datetime.hour >= 17 and local_datetime.minute >= 1:
                local_datetime = local_datetime+datetime.timedelta(days=1)
                local_datetime = local_datetime.replace(hour=4,minute=0,second=0,microsecond=0)

            #Skip weekends (Saturday and Sunday)
            elif local_datetime.weekday() == 5:
                local_datetime = local_datetime+datetime.timedelta(days=2)

            #Skip Sundays and holidays
            elif (local_datetime.weekday() == 6) or (local_datetime.date() in holidays):
                local_datetime = local_datetime+datetime.timedelta(days=1)
            
            else:
                #Add datetime to List and increase local date by one minute
                datetime_list.append(local_datetime)
                local_datetime = local_datetime+datetime.timedelta(minutes=1)

    #Create a DataFrame out of the datetime_list so we can merge prices data into it
    datetime_df = pd.DataFrame({'datetime': datetime_list})

    price_df = datetime_df.copy(deep=True)
    #Merge each of the ticker DataFrames with the datetime DataFrame
    for ticker in price_dict_df:
        #Keep all the datetimes in datetime_df (if there is no price data for a datetime, use the prices of a previous datetime)
        ticker_df = pd.merge_asof(datetime_df,price_dict_df[ticker],on='datetime')

        #Set the index to datetime and append the ticker name to the other columns
        ticker_df = ticker_df.set_index(['datetime'])
        ticker_df = ticker_df.add_suffix('_'+ticker)

        #Merge into main price_df
        price_df = price_df.merge(ticker_df,how='left',on='datetime')

    return price_df
    
    
def get_holidays(start_date,end_date):
    """
    CD 12/2020 Get holiday datetimes within start_date and end_date

    start_date - earlier datetime to get holiday dates from
    end_date - later datetime to get holiday dates to

    Return list of datetime holidays within start_date and end_date
    """
    holiday_list = []
    #Loop over each month between start and end date
    date = datetime.date(year=start_date.year,month=start_date.month,day=start_date.day)
    while date <= end_date:
        if date.month == 1:
            #New Years Day
            holiday_list.append(datetime.date(date.year,1,1))

            #Martin Luter King Day (3rd Monday of January)
            #Earliest is Jan 15th and latest is Jan 21st, number of days ahead the next Monday is from January 14th
            days_ahead = 7-datetime.date(date.year,1,14).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,1,14)+datetime.timedelta(days=days_ahead))
        elif date.month == 2:
            #Washington's Birthday (3rd Monday of February)
            #Earliest is Feb 15th and latest is Feb 21st, number of days ahead the next Monday is from February 14th
            days_ahead = 7-datetime.date(date.year,2,14).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,2,14)+datetime.timedelta(days=days_ahead))

        elif date.month == 3 or date.month==4:
            #Good Friday (Friday before Easter which falls on first Sunday following full moon on or after March 21) 
            holiday_list.append(calc_easter(date.year))

        elif date.month == 5:
            #Memorial Day (last Monday of May)
            #Earliest is May 25th and latest is May 31st, number of days ahead the next Monday is from May 24th
            days_ahead = 7-datetime.date(date.year,5,24).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,5,24)+datetime.timedelta(days=days_ahead))

        elif date.month == 7:
            #Independence Day
            if datetime.date(date.year,7,4).weekday() == 5:
                #Observed on Friday July 3rd
                holiday_list.append(datetime.date(date.year,7,3))
            elif datetime.date(date.year,7,4).weekday() == 6:
                #Observed on Monday July 5th
                holiday_list.append(datetime.date(date.year,7,5))
            else:
                holiday_list.append(datetime.date(date.year,7,4))
        
        elif date.month == 9:
            #Labor Day (1st Monday of September)
            #Earliest is Sep 1st and latest is Sep 6th, number of days ahead the next Monday is from August 31st
            days_ahead = 7-datetime.date(date.year,8,31).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,8,31)+datetime.timedelta(days=days_ahead))

        elif date.month == 11:
            #Thanksgiving Day (4th Thursday of November)
            #Earliest Monday before Thanksgiving is Nov 19th and latest is Nov 25th, add three extra days to make it Thursday (10 instead of 7)
            days_ahead = 10-datetime.date(date.year,11,18).weekday() #Weekday=0 for Monday
            holiday_list.append(datetime.date(date.year,11,18)+datetime.timedelta(days=days_ahead))

        elif date.month == 12:
            #Christmas Day
            if datetime.date(date.year,12,25).weekday() == 5:
                #Observed Friday December 24th
                holiday_list.append(datetime.date(date.year,12,24))
            elif datetime.date(date.year,12,25).weekday() == 6:
                #Observed Monday December 26th
                holiday_list.append(datetime.date(date.year,12,26))
            else:
                holiday_list.append(datetime.date(date.year,12,25))

        next_month = 1 if date.month==12 else date.month+1
        date = date.replace(month=next_month)

    #Remove any holidays that don't fall within range (can happen for first month or last month in loop)
    for date in holiday_list:
        if (date<start_date) or (date>end_date):
            holiday_list.remove(date)

    return holiday_list


def calc_easter(year):
    """
    Returns Easter as a date object.

    Obtained from https://code.activestate.com/recipes/576517-calculate-easter-western-given-a-year/?in=lang-python
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1    
    return datetime.date(year, month, day)


if __name__=="__main__":
    
    #Use these tickers to help guess the price of a chosen ticker. The first ticker must be the output ticker
    #tickers_input = ['NIO','TSLA','LI','XPEV','SPX','NDX','DJX','RUT']
    tickers_input = ['NIO','TSLA','SPX','NDX','DJX']

    #We are trying to guess the price of this stock. We assume the output ticker is also one of the inputs
    ticker_output = 'NIO'

    #Pull the saved data from stocks for a certain time period. CD 12/2020 Support a specific start_date and end_date
    start_date = datetime.date(2020,8,1)
    end_date = datetime.date(2020,11,30)

    (price_minute_df,price_daily_df) = extract_data(tickers_input,start_date,end_date)

    #Amount of time in the future that we are trying to guess the price. CD 12/2020 Suppport minutes and days in future
    predict_period = 1  #number of minutes or days in future
    predict_periodType = 'day' #only 'minute' or 'day' supported  

    features_list = features_list = ['close1min','close2min','close5min','close10min','close30min','close1hr','DiffHighLow','SMA5','SMA10','SMA20',\
            'SMA50','EMA5','EMA10','EMA20','EMA50','Boll20','Boll50','MACDline','SignalLine','MACDhist','RSI','Stochastic%K','Stochastic%K-%D']

    algorithm = 'Neural Network'

    stock_machine_learning(price_minute_df,price_daily_df,ticker_output,predict_period,predict_periodType,features_list,algorithm)


"""
Old way of getting input data
#Load the price data for each input ticker

ticker_index=1
for ticker in tickers_input:
    #Price history is a dictionary (ticker: price lists). First row is datetime and second row is the price
    temp_history = pickle.load(open('Price_History_' + ticker + '_' + start + '_' + end + '.p',"rb"))
    price_history = temp_history[ticker]
    
    if ticker == ticker_output:
        #For output ticker, save the datetime and price. We will align other tickers to same datetime and price. 
        datetime_data = np.array(price_history['datetime'])
        price_data[0] = np.array(price_history[1])
    else:
        #Align the price history data of this ticker with the price history data of the output ticker
        adjusted_price = align_data(datetime_data,price_history)
        price_data = np.append(price_data,[adjusted_price],axis=0)
        ticker_index=ticker_index+1   
"""

"""
#Convert timestamp to datetime
datetime_temp = []
for timestamp in price_daily_df.index:
    datetime_temp.append(datetime.datetime.fromtimestamp(timestamp/1000))
"""