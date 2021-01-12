# Stock_Exchange_Machine_Learning

More documentation and machine learning algorithms to come.

Start by creating a TD Ameritrade Developer Account (see my TD Ameritrade repository). Once you have an app setup, you will be able to save price history.

Open SavePriceHistory.py. Scroll to the bottom, edit the start and end dates to save price and volume history of tickers in ticker list to a pickle file. You can save "daily" price data or "minute" level price data.
  - If you want the files to be organized, you can edit file_organization and leave "tickers_list" blank. 
  
Open MainLearning.py. Scroll to the bottom, edit the start and end dates to be the same as the dates saved for the ticker. Please make sure to save daily and minute level price data before running this program and get daily price information as least 50 days before the start of the minute level data (in order to calculate technical indicators for all minute level data). Update "predict_period" and "predict_period_type" to make predictions at a specific time in the future. Currently, this allows for 'Linear Regression' or 'Neural Network'.
