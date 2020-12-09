"""
Saves minute level price history data for tickers from TD Ameritrade Website. 
Saves tickers in dataset dictionary and use the keys as the folder names. 
Manually update the start_dates_list and end_dates_list to get price history for that particular time range.
    Note: TD Ameritrade only allows access to minute level data for one or two months prior to the current date.
"""

from TDAmeritrade_API import get_access,get_price_history_lookback,get_price_history_dates
import time,datetime
import pickle

#Create a disctionary of tickers to get data from using the folder name to save it to as the key
dataset = {
    'Building': ['HD','LOW','MMM','PHG'],
    'Car': ['NIO','TM','TSLA','WKHS'],
    'Chips': ['AMD','IBM','INTC','NVDA'],
    'Climate': ['ECL','PHO','RSG','VEGN','WM'],
    'Computer': ['AAPL','HPQ','MSFT','MSI'],
    'Energy': ['JKS','NEE','REGI','SEDG','SPWR'],
    'Favs': ['GOOG','HXL','MTDR','NAIL'],
    'Food': ['CMG','COST','KR','UNFI'],
    'Health': ['CVS','JNJ','MRK','SYK'],
    'Indexes': ['DJX','NDX','SPX'],
    'Internet': ['AMTD','AMZN','FB','QCOM'],
    'Services': ['DFS','T','V','VZ']
}

#Get an access token in order to get price data
access_token=''
expire_time=0
[access_token,expire_time]=get_access(access_token,expire_time)
last_access = time.time()

#Loop through each subset of tickers
for key in dataset:

    #List of tickers to get and save into price history data
    tickers_list = dataset[key]

    #File path to save these tickers to
    file_path = 'C:/Coding_Practice/TD_Ameritrade/MachineLearning/Price_History_Monthly_{}/'.format(key)

    #Add each ticker to price history dictionary
    for ticker in tickers_list:

        #Price history is a dictionary that stores minute level data of each ticker.
        price_history = {} #Reset for each ticker (each file should only contain one ticker)
        price_history[ticker]=[]

        # *** USE START AND END DATE IF GETTING MINUTE LEVEL DATA ***
        #Get data at the minute level with each API call
        lookback = False
        frequencyType='minute'
        frequency=1

        month='Oct'
        #Get the last two months of data (I think two months is the max time TD Ameritrade will store minute level data)
        #The API will return minute level day for 10 day periods max, so make multiple calls of 10 day periods to get the full two months.
        start_dates_list = ['Thu Oct 01 01:00:00 2020','Mon Oct 12 01:00:00 2020','Mon Oct 26 01:00:00 2020']
        end_dates_list = ['Fri Oct 09 01:00:00 2020','Fri Oct 23 01:00:00 2020','Fri Oct 30 01:00:00 2020']


        # *** USE LOOKBACK DATE IF GETTING DAILY, WEEKLY, OR MONTHLY DATA ***
        #Get data at the minute level with each API call
        #lookback = True
        #frequencyType='daily'
        #frequency=1

        #Get data looking back a period of this length.
        periodType='year'
        period=20

        if lookback:
                
            #Create a price history dictionary. Each key contains a list of values (dates, prices, or volume)
            price_history[ticker]={'datetime': [], 'open': [], 'close': [], 'high': [], 'low': [], 'volume': []}
            price_history_temp = get_price_history_lookback(access_token,ticker,periodType,period,frequencyType,frequency)
            for price_info in price_history_temp['candles']:
                #Save the info in the dictionary 
                price_history[ticker]['datetime'].append(price_info['datetime'])
                price_history[ticker]['open'].append(price_info['open'])
                price_history[ticker]['close'].append(price_info['close'])
                price_history[ticker]['high'].append(price_info['high'])
                price_history[ticker]['low'].append(price_info['low'])
                price_history[ticker]['volume'].append(price_info['volume'])
        else:

            time_start=0
            #Create a price history dictionary. Each key contains a list of values (dates, prices, or volume)
            price_history[ticker]={'datetime': [], 'open': [], 'close': [], 'high': [], 'low': [], 'volume': []}

            for i in range(len(start_dates_list)):

                #Start date and end date since epoch (dates are used for determining days, weeks, months, or years - not specific to hours
                start_date = int(time.mktime(time.strptime(start_dates_list[i]))*1000)
                end_date = int(time.mktime(time.strptime(end_dates_list[i]))*1000)

                time_end = time.time() #time_end - time_start = time since last API call
                if time_end-time_start < 30:
                    #Only make 2 API calls per second
                    time.sleep(30-(time_end-time_start))

                #Refresh access token when we reach 27 minutes (it expires after 30)
                if time.time()-last_access > 27*60:
                    [access_token,expire_time]=get_access()
                    last_access = time.time()

                time_start = time.time()
                #price_history = get_price_history_lookback(access_token,ticker,periodType,period,frequencyType,frequency)
                price_history_temp = get_price_history_dates(access_token,ticker,start_date,end_date,frequencyType,frequency)

                for price_info in price_history_temp['candles']:
                    #Save the info in the dictionary 
                    price_history[ticker]['datetime'].append(price_info['datetime'])
                    price_history[ticker]['open'].append(price_info['open'])
                    price_history[ticker]['close'].append(price_info['close'])
                    price_history[ticker]['high'].append(price_info['high'])
                    price_history[ticker]['low'].append(price_info['low'])
                    price_history[ticker]['volume'].append(price_info['volume'])


        #Label the file with the time and ticker
        if lookback:
            file_name = 'Price_History_{}'.format(ticker) + '_{}'.format(period) + periodType + 's.p'
        else:
            #Use start_str and end_str if we are grabbing more or less than a particular month of data
            #start_str = start_dates_list[0][4:7] + start_dates_list[0][8:10]
            #end_str = end_dates_list[-1][4:7] + end_dates_list[-1][8:10]
            #file_name = 'Price_History_' + '-'.join(tickers_list) + '_' + start_str + '_' + end_str + '.p'

            file_name = 'Price_History_{}'.format(ticker) + '_' + month + '.p'
        pickle.dump(price_history,open(file_path + file_name,"wb"))


# datetime.datetime.fromtimestamp(price_history[ticker]['datetime'][0]/1000)