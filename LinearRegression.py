
#This uses linear regression to guess future stock prices

import numpy as np
import matplotlib.pyplot as plt
import datetime,math,pickle
from StockFeatures import add_features


if __name__=="__main__":

    #Pull the saved data from stocks for a certain time period
    month='Aug'
    file_path = 'C:/Coding_Practice/TD_Ameritrade/MachineLearning/Price_History_Monthly_Car/'
    price_data={}
    price_data_daily = {}

    #Use these tickers to help guess the price of a chosen ticker. The first ticker must be the output ticker
    #tickers_input = ['NIO','TSLA','LI','XPEV','SPX','NDX','DJX','RUT']
    tickers_input = ['NIO','LI']

    #We are trying to guess the price of this stock. We assume the output ticker is also one of the inputs
    ticker_output = 'NIO'

    #Approximate number of minutes in the future that we are trying to guess the price
    predict_time = 10

    ticker_index=1    

    datetime_begin = 0
    datetime_end = 10**20
    #We have price_data for each ticker
    for ticker in tickers_input:
        temp_history=pickle.load(open(file_path + 'Price_History_' + ticker + '_' + month + '.p',"rb"))
        price_data.update(temp_history)

        #Save the latest begin time so each stock begins at the same time
        datetime_begin = max(datetime_begin,price_data[ticker]['datetime'][0])

        #Save the earliest end time so each stock ends at the same time
        datetime_end = min(datetime_end,price_data[ticker]['datetime'][-1])

    #Get the daily price_data for each ticker for the past 6 months and turn into a numpy array
    temp_history=pickle.load(open(file_path + 'Price_History_' + ticker_output + '_6months_Aug.p',"rb"))
    price_data_daily.update(temp_history)
    for key in temp_history[ticker_output]:
        price_data_daily[ticker_output][key] = np.array(temp_history[ticker_output][key])

    #Make all stocks use same time range.
    timestamp_data = []
    dttm_data = []
    for timestamp in range(datetime_begin,datetime_end,60000):
        local_date = datetime.datetime.fromtimestamp(timestamp/1000)

        #Only record values during trading hours
        if (local_date.hour>6 or (local_date.hour==6 and local_date.minute>29)) and (local_date.hour<13 or (local_date.hour==13 and local_date.minute==0)): 
            #Only include weekdays and days that are not holidays (July 3)
            if local_date.weekday()<=4 and (local_date.month != 7 or local_date.day !=3):
                timestamp_data = np.append(timestamp_data,timestamp)
                dttm_data = np.append(dttm_data,local_date)

    data = {}
    #Create minute level data from start to finish for each ticker
    for ticker in tickers_input:
        j=0
        data[ticker]={'open': [], 'close': [], 'high': [], 'low': [], 'volume': []}
        #Loop through every datetime from start to finish. (Repeat data from earlier time if none recorded already at dttm)
        for timestamp in timestamp_data:

            #Keep looping though the ticker data datetimes until we find one greater than timestamp
            while j<len(price_data[ticker]['datetime'])-1 and price_data[ticker]['datetime'][j]<timestamp:
                j=j+1

            #Record data for dttm as close as we can get to dttm but still recorded before dttm
            for key in data[ticker]:
                #Convert to a numpy array
                data[ticker][key] = np.append(data[ticker][key],price_data[ticker][key][j-1])
        
    predict_time = 30 #Predict 30 minutes into the future
    features_list = [1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] #Use price data as features
    lamb = 1000 #Regularization parameter used to prevent overuse of a few features
    (output_data,input_data,initial_data) = add_features(dttm_data,data,price_data_daily,ticker_output,predict_time,features_list)

    #Construct the input and output matrix. 
    rows = input_data.shape[0] #Number of features
    cols = input_data.shape[1] #Number of training samples
    #First half of data to train on
    X1 = np.transpose(input_data[0:rows,0:math.floor(cols/2)])
    y1 = output_data[0:math.floor(cols/2)]
    #Second half of data to test on
    X2 = np.transpose(input_data[0:rows,math.floor(cols/2):cols])
    y2 = output_data[math.floor(cols/2):cols]

    #Use regularization 
    L = np.identity(rows)
    L[0][0]=0

    #Normal Equation for Linear Regression with Regularization: theta = (X^T*X+lambda*L)^-1*X^T*y
    X_T = np.transpose(X1)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X1)+lamb*L),X_T),y1)

    temp1 = np.dot(X1,theta)-y1
    cost1 = np.dot(np.transpose(temp1),temp1)/(2*cols)*100 + lamb*sum(np.square(theta[1:]))

    temp2 = np.dot(X2,theta)-y2
    cost2 = np.dot(np.transpose(temp2),temp2)/(2*cols)*100 + lamb*sum(np.square(theta[1:]))

    #Make sure the output prediction is not too similar to the input
    initial_data1 = initial_data[0:math.floor(cols/2)]
    initial_data2 = initial_data[math.floor(cols/2):cols]
    temp_input = np.dot(X2,theta)-initial_data[math.floor(cols/2):cols]
    cost_input = np.dot(np.transpose(temp_input),temp_input)/(2*cols)*100

    #Predict the amount of money we would make if we bought when the predicted output hit a certain threshold
    expect = np.dot(X2,theta)
    threshold = 0.5
    gain=0
    loss=0
    times_above=0
    times_below=0
    for i in range(len(expect)):
        if expect[i]-initial_data2[i]>threshold:
            if y2[i]-initial_data2[i]<0:
                loss = loss-(y2[i]-initial_data2[i])
                times_below = times_below+1
            else:
                gain = gain + y2[i]-initial_data2[i]
                times_above=times_above+1

    #Take off taxes from profit
    profit=(gain*.8-loss)*5
    TD_loss = (times_above+times_below)*.1

    #Plot expected vs. actual results
    plt.plot(expect,label="Expected")
    plt.plot(y2,label="Output")
    plt.plot(initial_data2,label="Input")
    plt.legend()
    plt.show()

    test=1
#datetime.datetime.fromtimestamp(datetime_data[91]/1000)

"""
Code to estimate profit
profit=0
times_above=0
for i in range(len(expect)):
    if expect[i]-X2[i]>1:
        profit = profit + y2[i]-X2[i]
        times_above=times_above+1
"""


"""
plt.plot(expect,label="Expected")
plt.plot(y2,label="Output")
plt.plot(initial_data2,label="Input")
plt.legend()
plt.show()


plt.plot(np.dot(X1,theta),label="Expected")
plt.plot(y1,label="Output")
plt.plot(initial_data1,label="Input")
plt.legend()
plt.show()
"""

"""
fig, ax1 = plt.subplots()
ax1.plot(data[ticker]['high'],label="High")
ax1.plot(data[ticker]['low'],label="Low")
ax2 = ax1.twinx()
ax2.plot(data[ticker]['volume'],label="volume")
plt.legend()
plt.show()

plt.plot(data[ticker]['volume'])
plt.show()
"""


def align_data(datetime_data,price_history):
    """
    Many times the saved prices recorded by TD Ameritrade are recorded at slightly different times.
    Ex: Getting the minute level prices of MSFT and AAPL between June 1st and June 2nd will not return the same number of prices nor the same price timings.
        Some minutes are skipped and therefore it throws of the indexes

    This function approximately aligns the times/prices in price_history to the datetimes in datetime_data, so they are the same length for future computations.
    datetime_data - numpy array of datetimes to align to
    price_history - list of lists (first list contains datetimes and second list contains prices)

    Returns the prices aligned to datetime_data
    """
    adjusted_price=[]
    j=1 #Index for price_history of ticker
    for i in range(len(datetime_data)):
        if price_history[0][j]==datetime_data[i]:
            adjusted_price.append(price_history[1][j])

        #If the datetime of price history is less than the datetime_data, increase ticker index until we have a similar enough date
        elif price_history[0][j]<datetime_data[i]:
            while j<len(price_history[0])-1 and price_history[0][j]<datetime_data[i]:
                j=j+1

            #Find closest two datetimes and approximate the price of ticker to be at the same time as the datetime_data
            if abs(datetime_data[i]-price_history[0][j])>abs(datetime_data[i]-price_history[0][j-1]):
                adjusted_price.append(price_history[1][j-1])
            else:
                adjusted_price.append(price_history[1][j])

        #If the datetime of the ticker is greater than datetime_data, decrease ticker index until we have a similar enough date
        else:
            while j>0 and price_history[0][j]<datetime_data[i]:
                j=j-1
            
            #Find closest two datetimes and approximate the price of ticker to be at the same time as datetime_data
            if abs(datetime_data[i]-price_history[0][j])>abs(datetime_data[i]-price_history[0][j+1]):
                adjusted_price.append(price_history[1][j+1])
            else:
                adjusted_price.append(price_history[1][j])

    return adjusted_price
    

"""
Old way of getting input data
#Load the price data for each input ticker
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