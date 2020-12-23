"""
Linear Regression functions. Takes in pandas DataFrames for input and output variables and returns the parameters and predicted output.

CD 12/2020 - Create a linear_reg_normal function (linear regression using normal equation to solve)
           - Use DataFrames instead of numpy arrays
           - Remove code to load price data (done in Main_Learning) and update linear regression to use pandas dataframes
           - Move code on predicting profit to EvaluatingLearningAlgorithms.py

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime,math,pickle
from StockFeatures import add_features


def linear_reg_normal(input_data_df,output_data_df,lamb):
    """
    Performs linear regression using the normal function method (instead of a optimization function).
    Takes in input and output DataFrames indexed by Datetime and returns the predicted output and parameters.

    input_data_df - input data indexed by Datetime. Each column is a new feature.
    output_data_df - output data indexed by Datetime. Should contain only one column of data.
    lamb - lamda regularization parameter

    Returns (theta,predicted_output)
        theta - parameters to the linear regression algorithm
        predicted_output - the prediction of the linear regression algorithm
    """

    #Construct the input and output matrix. 
    #num_samples = input_data_df.shape[0] #Number of training samples
    num_features = input_data_df.shape[1] #Number of features
    
    #Input and output parameters for linear regression
    X = input_data_df.copy(deep=True)
    y = output_data_df.copy(deep=True)

    #Use regularization 
    L = np.identity(num_features)
    L[0][0]=0 #Bias parameter does not need regularization

    #Normal Equation for Linear Regression with Regularization: theta = (X^T*X+lambda*L)^-1*X^T*y
    X_T = np.transpose(X)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X)+lamb*L),X_T),y)

    predicted_output = np.dot(X,theta)

    return (theta,predicted_output)

    
"""

#Plot expected vs. actual results
plt.plot(predicted_output,label="Expected")
plt.plot(y,label="Output")
plt.plot(initial_data2,label="Input")
plt.legend()
plt.show()

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
