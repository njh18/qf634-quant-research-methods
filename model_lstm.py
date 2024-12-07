"""
LSTM Model Training using price of stocks 
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Input
from keras_tuner.tuners import RandomSearch


# split the data into training and testing
def split_train(split, X, y):
  split = int(split * len(X))
  X_train = X[:split]
  Y_train = y[:split]
  return X_train, Y_train

def split_test(split, X, y):
  split = int(split * len(X))
  X_test = X[split:]
  Y_test = y[split:]
  return X_test, Y_test

#define main function to run the model
def main(filtered_data):

  results = {}

  for stock in filtered_data.columns:
    print(f"Training model for stock: {stock}")
    X = np.asarray([filtered_data.iloc[i - 1:i, filtered_data.columns.get_loc(stock)].values for i in range(1, len(filtered_data))])
    y = np.asarray([filtered_data.iloc[i, filtered_data.columns.get_loc(stock)] for i in range(1, len(filtered_data))])

    # split 80-20
    split = 0.8

    X_train, y_train = split_train(split, X, y)
    X_test, y_test = split_test(split, X, y)
    print(f'X_train: {X_train.shape} | Y_train: {y_train.shape}')
    print(f'X_test: {X_test.shape}   | Y_test: {y_test.shape}')

    ## feature scaling 
    # split into X and Y to prevent data leakage 
    scaler_X = MinMaxScaler(feature_range=(0,1))
    scaler_Y = MinMaxScaler(feature_range=(0,1))

    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(y_train.reshape(-1, 1))

    X_test_scaled = scaler_X.transform(X_test)
    Y_test_scaled = scaler_Y.transform(y_test.reshape(-1, 1))

    # reshape into 3D for for lstm model
    X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    ## lstm model building
    # initialize RNN
    model = Sequential()
    # first layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Dropout(0.2))
    # second layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # third layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # fourth layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(1))
    # compile RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    ## train lstm model on training set using RNN
    model.fit(X_train_scaled, Y_train_scaled, epochs=100, batch_size=10)

    model.summary()

    predict_train_model = model.predict(X_train_scaled)
    predict_test_model = model.predict(X_test_scaled)

    predict_train_model=predict_train_model[:, 0]
    predict_test_model=predict_test_model[:, 0]

    print(predict_train_model.shape)
    print(predict_test_model.shape)

    predict_stock_price = model.predict(X_test_scaled)
    predict_stock_price = predict_stock_price[:, 0]

    pred_test_price = scaler_Y.inverse_transform(predict_stock_price.reshape(-1, 1))
    actual_test_price = scaler_Y.inverse_transform(Y_test_scaled)
    test_rmse = math.sqrt(mean_squared_error(actual_test_price, pred_test_price))

    pred_train_price = scaler_Y.inverse_transform(predict_train_model.reshape(-1, 1))
    actual_train_price = scaler_Y.inverse_transform(Y_train_scaled)
    train_rmse = math.sqrt(mean_squared_error(actual_train_price, pred_train_price))
    print(test_rmse)
    print(train_rmse)
    results[stock] = {
          'stock': stock,
          'train_rmse': train_rmse,
          'test_rmse': test_rmse,
          'pred_test_price': pred_test_price,
          'actual_test_price': actual_test_price
      }
  return results

if __name__ == '__main__':
  ## data preparation 
  data = pd.read_csv('df_price.csv')
  print(data.head())
  # set date range
  filtered_data = data[(data['Date'] > '2021-01-01') 
      & (data['Date'] < '2022-12-31')]

  filtered_data = filtered_data.dropna()

  filtered_data = filtered_data.drop(columns=['Date'])
  model_results = main(filtered_data)
  print(model_results)

  # Number of stocks
  num_stocks = len(model_results)

  # Create a figure with subplots
  fig, axes = plt.subplots(nrows=num_stocks, ncols=1, figsize=(10, num_stocks * 4))
  fig.tight_layout(pad=5.0)

  # Plot each stock's actual and predicted prices in a subplot
  for i, (stock, data) in enumerate(model_results.items()):
      ax = axes[i] if num_stocks > 1 else axes  # Handle single subplot case
      ax.plot(data['actual_test_price'], label=f"{stock} Actual Price", linestyle='--')
      ax.plot(data['pred_test_price'], label=f"{stock} Predicted Price", linestyle='-')
      ax.set_title(f"{stock} Price Prediction (2021-2022)")
      ax.set_xlabel('Time')
      ax.set_ylabel('Price in $')
      ax.legend()

  # Show the plot
  plt.show()
'''
  plt.plot(actual_test_price, label="Actual Price",linestyle='--') 
  plt.plot(pred_test_price,label="Predicted Price",linestyle='-')
  plt.title('Stock Price Prediction from 2021 to 2022')
  plt.xlabel('Time')
  plt.ylabel('Price in $')
  plt.legend()
  plt.show()
'''
