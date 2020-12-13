import numpy as np
import datetime


def calc_sma(time_data,price_data,daily_close_dttm,daily_close_prices,interval):
    """
    Calculate the simple moving average for each time data point using the daily closing prices and the last available price
    The interval represents the number of days to use in the average
    The daily closing prices must have data for at least interval days before the time data starts.
     For example: if we are calculating the 20 day simple moving average (interavl = 20), the daily_close values must start 20 days before the time_data.
                  Otherwise, we would not be able to calculate the SMA for the first data point in time data.
                  SMA = (sum(close price on previous 19 days) + current_price)/20

    time_data - the datetimes that we want to calculate the simple moving average for
    price_data - the prices corresponding to those time values
    daily_close_dttm - contains a list of timestamps (each index of this list matches with an index in the daily_close_prices list)
    daily_close_prices - contains a list of close prices of a stock 
    interval - the number of days to include in the simple moving average

    returns the simple moving average

    """

    daily_index_start=0
    #Find the index in the daily close prices where the time_data starts
    for daily_index in range(len(daily_close_dttm)):
        if (daily_close_dttm[daily_index]>=time_data[0] and daily_index_start==0):
            daily_index_start = daily_index

    #Calculate the simple moving average
    sma = []
    days = 0 
    local_date = time_data[0]
    for dttm_index in range(len(time_data)):
        local_date_prev = local_date
        local_date = time_data[dttm_index]

        #Increase day if this date is one day ahead of previous date (if they are different)
        if local_date.day != local_date_prev.day:
            days = days+1

        #Sum up the close prices for the (interval) previous days and the price at the dttm_index
        sma = np.append(sma,(sum(daily_close_prices[daily_index_start-interval+days+1:daily_index_start+days])+price_data[dttm_index])/interval)


    return sma


def calc_ema(time_data,price_data,daily_close_dttm,daily_close_prices,interval):
    """
    Calculate the exponential moving average for each time data point using the daily closing prices and the previous EMA
    The interval represents the number of days to use in the average.
    The daily closing prices must have data for at least 2*interval days before the time data starts.
     For example: if we are calculating the 20 day exponential moving average (interavl = 20), the daily_close values must start 40 days before the time_data.
                  Otherwise, we would not be able to calculate the EMA for the first data point in time data.
                  EMA = (Current_Price or Close_Price) * Multiplier + EMA_yesterday * (1-multiplier)
                  multiplier = 2/(1+interval)
                  first EMA = 20 day Exponential Moving Average day using only close prices

                  Basically we are calculating two interval day EMA's. The first one uses only close prices of the 2*interval-2 to interval-1 previous days.
                  The second uses the first EMA as the start point and then the interval-2 previous day close prices to the current price.

    time_data - the datetimes that we want to calculate the exponential moving average for
    price_data - the prices corresponding to those time values
    daily_close_dttm - contains a list of timestamps (each index of this list matches with an index in the daily_close_prices list)
    daily_close_prices - contains a list of close prices of a stock 
    interval - the number of days to include in the exponential moving average

    Returns 
        Exponential moving average (EMA) based on time_data
        Daily exponential moving average based on daily_close_dttm

    """

    daily_index_start=0
    daily_ema = []
    mult = 2/(1+interval) #Multiplier
    #Find the index in the daily close prices where the time_data starts
    for daily_index in range(len(daily_close_dttm)):
        if (daily_close_dttm[daily_index]>=time_data[0] and daily_index_start==0):
            daily_index_start = daily_index

        #Calculate the interval day EMA for all daily values available.
        if daily_index == 0:
            daily_ema = np.append(daily_ema,daily_close_prices[0]) #For the first EMA just set to the close price
        else:
            daily_ema = np.append(daily_ema,daily_close_prices[daily_index] * mult + daily_ema[daily_index-1] * (1-mult))

    ema = []
    days = 0 
    local_date = time_data[0]
    #Calculate the EMA for the values
    for dttm_index in range(len(time_data)):
        local_date_prev = local_date
        local_date = time_data[dttm_index]
        if dttm_index < len(time_data)-1:
            local_date_next = time_data[dttm_index+1]

        #Increase day if this date is one day ahead of previous date (if they are different)
        if local_date.day != local_date_prev.day:
            days = days+1

        if local_date_next.day != local_date.day:
            stop=1
        
        ema_temp = price_data[dttm_index] * mult + daily_ema[daily_index_start+days-1] * (1-mult)

        #Add the new second interval day EMA to our array
        ema = np.append(ema,ema_temp)

    return (ema,daily_ema)




def calc_std(time_data,price_data,daily_close_dttm,daily_close_prices,sma,interval):
    """
    Calculate the standard deviation for each time data point using the daily closing prices and the last available price
    The interval represents the number of days to use in calculating the standard deviation
    The daily closing prices must have data for at least interval days before the time data starts.
     For example: if we are calculating the 20 day standard deviation (interavl = 20), the daily_close values must start 20 days before the time_data.
                  Otherwise, we would not be able to calculate the std for the first data point in time data.
                  std = (sum((close_price - SMA)^2)/20)^1/2  
                   - we sum up the square difference between close price and sma except for the current day (we use the current price if close price is not available yet)


    time_data - the datetimes that we want to calculate the standard deviation for
    price_data - the prices corresponding to those time values
    daily_close_dttm - contains a list of datetimes (each index of this list matches with an index in the daily_close_prices list)
    daily_close_prices - contains a list of close prices of a stock 
    sma - the simple moving average of the stock over the same interval, the sma is calculated for each of the time_data points
        (Ex: if interval = 20, sma should be the 20 day simple moving average)
    interval - the number of days to include in the standard deviation

    returns the standard deviation

    """

    daily_index_start=0
    #Find the index in the daily close prices where the time_data starts
    for daily_index in range(len(daily_close_dttm)):
        if (daily_close_dttm[daily_index]>=time_data[0] and daily_index_start==0):
            daily_index_start = daily_index

    #Calculate the standard deviation
    std = []
    days = 0 
    local_date_prev = time_data[0]
    for dttm_index in range(len(time_data)):
        #Increase day if this date is one day ahead of previous date (if they are different)
        if time_data[dttm_index].day != local_date_prev.day:
            days = days+1

        #Sum up the squared differences of close price and the SMA for the (interval-1) previous days 
        squared_diff = sum(np.square(daily_close_prices[daily_index_start-interval+days+1:daily_index_start+days]-sma[dttm_index]))

        #Add in the square difference of the current price and the SMA
        squared_diff = squared_diff + (price_data[dttm_index]-sma[dttm_index])**2

        #Append the standard deviation for this time point
        std = np.append(std,(squared_diff/interval)**0.5)

        local_date_prev = time_data[dttm_index]

    return std


def calc_macd(time_data,price_data,daily_close_dttm,daily_close_prices):
    """
    Calculate the moving average convergence/divergence oscillator (MACD) for each time data point.
    The daily closing prices must have data for at least interval days before the time data starts.

    time_data - the datetimes that we want to calculate the macd for
    price_data - the prices corresponding to those time values
    daily_close_dttm - contains a list of timestamps (each index of this list matches with an index in the daily_close_prices list)
    daily_close_prices - contains a list of close prices of the stock 

    Returns the MACD line, the signal line, and the MACD histogram
        macd_line = EMA_12days - EMA_26days (using close prices in price_data and daily_close prices)
        signal_line = EMA_9days (using the macd line)
        macd_histogram = macd_line - signal_line

    """
    #The MACD line is the 12 day exponential moving average minus the 26 day exponential moving average
    (ema12,daily_ema12) = calc_ema(time_data,price_data,daily_close_dttm,daily_close_prices,12)
    (ema26,daily_ema26) = calc_ema(time_data,price_data,daily_close_dttm,daily_close_prices,26)
    macd_line = ema12-ema26
    daily_macd_line = daily_ema12 - daily_ema26

    #The signal line is the 9 day exponential moving average of the MACD line
    (signal_line,_) = calc_ema(time_data,macd_line,daily_close_dttm,daily_macd_line,9)

    #The MACD histogram is the macd_line minus the signal_line
    macd_histogram = macd_line - signal_line

    return (macd_line,signal_line,macd_histogram)



def calc_rsi(time_data,price_data,daily_close_dttm,daily_close_prices):
    """
    Calculate the Relative Stregnth Index (RSI) for each time data point using the daily closing prices and the last available price.
    The daily closing prices must have data for at least 14 days before the time data starts.
    
    time_data - the datetimes that we want to calculate the RSI for
    price_data - the prices corresponding to those time values
    daily_close_dttm - contains a list of timestamps (each index of this list matches with an index in the daily_close_prices list)
    daily_close_prices - contains a list of close prices of the stock 

    Returns the RSI for each point using the time_data
        We caculate the RSI in two parts:
        RSI_First = 100 - 100/(1+average_gain/average_loss)
            average_gain = (sum of the gains between closing prices over 14 days)/14  - ignore days that closed lower than the previous day
            average_loss = (sum of the losses between clsoing prices over 14 days)/14 - ignore days that closed higher than the previous day
        
        RSI = 100 - 100/(1+(average_gain_prev*13+current_gain)/(average_loss_prev*13+current_loss))

    """

    daily_index_start=0
    average_gain = 0
    average_loss = 0
    daily_rsi = []
    daily_average_gain = []
    daily_average_loss = []
    #Find the index in the daily close prices where the time_data starts
    for daily_index in range(len(daily_close_dttm)):
        if (daily_close_dttm[daily_index]>=time_data[0] and daily_index_start==0):
            daily_index_start = daily_index

        #Part 1 of RSI calculation: get the average gain/loss over the past daily_index days
        if daily_index > 0 and daily_index < 15:
            #Difference between new close and previous day's close price
            close_diff = daily_close_prices[daily_index]-daily_close_prices[daily_index-1]

            #Add the new gain to average gain if price closed higher than previous day
            if close_diff>0:
                average_gain = (average_gain*(daily_index-1)+close_diff)/daily_index
                average_loss = average_loss*(daily_index-1)/daily_index
            #Add the new loss to average loss if price closed higher than previous day
            elif close_diff<0:
                average_gain = average_gain*(daily_index-1)/daily_index
                average_loss = (average_loss*(daily_index-1)-close_diff)/daily_index
            #The price closed the same
            else:
                average_gain = average_gain*(daily_index-1)/daily_index
                average_loss = average_loss*(daily_index-1)/daily_index
            
            
        #Part 2 of RSI calculation: average gain/loss = (average gain/loss previous * 13 + new gain/loss)/14
        elif daily_index >= 15:
            #Difference between new close and previous day's close price
            close_diff = daily_close_prices[daily_index]-daily_close_prices[daily_index-1]

            if close_diff > 0:
                average_gain = (average_gain*13 + close_diff)/14
                average_loss = average_loss*13/14
            elif close_diff < 0:
                average_gain = average_gain*13/14
                average_loss = (average_loss*13 - close_diff)/14
            else:
                average_gain = average_gain*13/14
                average_loss = average_loss*13/14

        #Save the average_gain, average_loss, and RSI daily values
        daily_average_gain = np.append(daily_average_gain,average_gain)
        daily_average_loss = np.append(daily_average_loss,average_loss)
        if average_loss == 0:
            #Set RSI to 50 for now until we get an average loss recording (can't divide by zero)
            daily_rsi = np.append(daily_rsi,50)
        else:
            daily_rsi = np.append(daily_rsi,100-100/(1+average_gain/average_loss))


    #Calculate the RSI for time_data
    rsi = []
    days = 0 
    local_date = time_data[0]
    for dttm_index in range(len(time_data)):
        local_date_prev = local_date
        local_date = time_data[dttm_index]

        #Increase day if this date is one day ahead of previous date
        if local_date.day != local_date_prev.day:
            days = days+1
        
        close_diff = price_data[dttm_index] - daily_close_prices[daily_index_start+days-1]
        if close_diff > 0:
            average_gain = (daily_average_gain[daily_index_start+days-1]*13 + close_diff)/14
            average_loss = daily_average_loss[daily_index_start+days-1]*13/14
        elif close_diff < 0:
            average_gain = daily_average_gain[daily_index_start+days-1]*13/14
            average_loss = (daily_average_loss[daily_index_start+days-1]*13 - close_diff)/14
        else: #No change in price from previous close price
            average_gain = daily_average_gain[daily_index_start+days-1]*13/14
            average_loss = daily_average_loss[daily_index_start+days-1]*13/14

        #Add the RSI to our array
        rsi = np.append(rsi,100-100/(1+average_gain/average_loss))


    return rsi


def calc_stochastic_oscillator(time_data,price_data,daily_dttm,daily_high_prices,daily_low_prices,daily_close_prices):
    """
    Calculate the Stochastic Oscillator for each time data point using the daily high/low prices and the last available price.
    The daily high/low prices must have data for at least 14 days before the time data starts.
    
    time_data - the datetimes that we want to calculate the stochastic oscillator for
    price_data - the prices corresponding to those time values
    daily_dttm - contains a list of timestamps (each index of this list matches with an index in the daily_close_prices list)
    daily_high_prices - contains a list of the daily high prices of a stock
    daily_low_prices - contains a list of the daily low prices of a stock
    daily_close_prices - contains a list of the daily close prices of a stock

    Returns the K_line and D_line of the stochastic oscillator
        K_line = %K = 100*(Current_Price - 14 day Low)/(14 day High - 14 day Low)
        D_line = %D = 100*(Current_Price - 14 day Low)/(14 day High - 14 day Low)
    """

    daily_index_start=0
    high_13day = []
    low_13day = []
    daily_k_line = []
    for daily_index in range(len(daily_dttm)):
        #Find the index in the daily high/low prices where time_data starts
        if (daily_dttm[daily_index]>=time_data[0] and daily_index_start==0):
            daily_index_start = daily_index

        #Find the 13 day highs and lows for each daily index
        if daily_index < 13:
            high_13day = np.append(high_13day,np.amax(daily_high_prices[:daily_index+1]))
            low_13day = np.append(low_13day,np.amin(daily_low_prices[:daily_index+1]))
                
        elif daily_index >= 13:
            high_13day = np.append(high_13day,np.amax(daily_high_prices[(daily_index-12):daily_index+1]))
            low_13day = np.append(low_13day,np.amin(daily_low_prices[(daily_index-12):daily_index+1]))

        daily_k_line = np.append(daily_k_line,100*(daily_close_prices[daily_index]-low_13day[daily_index])/(high_13day[daily_index]-low_13day[daily_index]))


    #Calculate the K_line and D_line for time_data
    k_line = []
    d_line = []
    days = 0 
    day_index_start = 0 #The start index of the most recent day
    local_date = time_data[0]
    for dttm_index in range(len(time_data)):
        local_date_prev = local_date
        local_date = time_data[dttm_index]

        #Increase day if this date is one day ahead of previous date
        if local_date.day != local_date_prev.day:
            days = days+1
            day_index_start = dttm_index

        #Find the 14 day max and min using the most recent price data
        high_14day = max(high_13day[daily_index_start+days-1],max(price_data[day_index_start:dttm_index+1]))
        low_14day = min(low_13day[daily_index_start+days-1],min(price_data[day_index_start:dttm_index+1]))

        k_line = np.append(k_line,100*(price_data[dttm_index]-low_14day)/(high_14day-low_14day))
        d_line = np.append(d_line,(k_line[-1]+daily_k_line[daily_index_start+days-1]+daily_k_line[daily_index_start+days-2])/3)

    return k_line,d_line