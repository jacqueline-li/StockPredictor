# -*- coding: utf-8 -*-
"""
jupyter notebook available at: https://colab.research.google.com/drive/1Iss00IC6cEtZQLEpAQzJjKDIXxNAOwDk
"""

# With Tesla's recent boom in stock prices, this program uses an artificial recurrent neural network (Long Short Term Memory/LSTM)
# to predict the closing stock price of Tesla using the past 60 day's stock price.

# Import libraries 
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Keras import
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Get stock quote
df = web.DataReader('TSLA', data_source="yahoo", start="2018-02-01", end='2020-02-04')
# Show data
df

# Get the number of rows and columns in the data set
df.shape

# Visualize closing price history
plt.figure(figsize=(16,8))
plt.title('Closing Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=16)
plt.ylabel('Closing Price USD ($)', fontsize=16)
plt.show()

# Create new dataframe with only the 'Close' column to get closing values
data = df.filter(['Close'])
# Convert dataframe to numpy array
dataset = data.values
# Get number of rows to train model with
training_data_len = math.ceil(len(dataset) * .8)

training_data_len

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scale_data = scaler.fit_transform(dataset)

scale_data

# Create training data set

# Create scaled training data set
train_data = scale_data[0:training_data_len, :]
# Split data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])