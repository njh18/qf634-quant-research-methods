"""
LSTM Model Training
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Input
from keras_tuner.tuners import RandomSearch

## data preparation 
# read pickle file
data = pd.read_pickle('df_pct_change.pkl')
print(data.head())

data = data.drop(columns=['Date'])
# split data into training and testing
train_ratio = 0.8
train_size = int(train_ratio * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

X_train = []
y_train = []

Time_steps = 60 # 2 months 

for i in range(Time_steps, len(train_scaled)):
  X_train.append(train_scaled[i-Time_steps:i, 0])
  y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# reshape data for lstm model - 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

## lstm model building
# initialize RNN
model = Sequential()
# first layer 
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# second layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
# third layer 
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
# fourth layer 
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
# output layer
model.add(Dense(1))
# compile RNN 
model.compile(optimizer='adam', loss='mean_squared_error')






# # split the data into training and testing
# def split_train(split, X, y):
#   split = int(split * len(X))
#   X_train = X[:split]
#   Y_train = y[:split]
#   return X_train, Y_train

# def split_test(split, X, y):
#   split = int(split * len(X))
#   X_test = X[split:]
#   Y_test = y[split:]
#   return X_test, Y_test

# X = np.asarray([data.iloc[i - 1:i, 0].values for i in range(1, len(data))])
# y = np.asarray([data.iloc[i, 0] for i in range(1, len(data))])

# split = 0.8

# X_train, y_train = split_train(split, X, y)
# X_test, y_test = split_test(split, X, y)
# # print(f'X_train: {X_train.shape} | Y_train: {y_train.shape}')
# # print(f'X_test: {X_test.shape}   | Y_test: {y_test.shape}')

# ## reshaping the data for lstm model - 3D   
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# # print(f'X_train: {X_train.shape} | Y_train: {y_train.shape}')
# # print(f'X_test: {X_test.shape}   | Y_test: {y_test.shape}')

# ## lstm model building
# def build_model(hp):
#     model = Sequential()
#     #TODO : add layers to the model?
#     model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
#     model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
#     model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False))
#     model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
#     model.add(Dense(1))
#     model.add(Activation('linear'))
#     model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
#                   loss='mean_squared_error')
#     return model

# ## random search for optimal hyperparameters values 
# tuner = RandomSearch(
#     build_model,
#     objective='val_loss',
#     max_trials=5,
#     executions_per_trial=3,
#     directory='lstm',
#     project_name='stock_price_prediction'
# )

# tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# best_hyperparam = tuner.get_best_hyperparameters(num_trials=1)[0]
# # print(best_hyperparam.values)
# # print(f"Best neurons: {best_hyperparam.get('units')}")
# # print(f"Best drop_rate: {best_hyperparam.get('dropout')}")
# # print(f"Best learning rate: {best_hyperparam.get('learning_rate')}")

# tuner.results_summary()

# lstm_model = build_model(best_hyperparam)

# ## train lstm model on training set 
# history = lstm_model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# predicted_stock_price = lstm_model.predict(X)
# # print(predicted_stock_price)

# pred_val = [predicted_stock_price[i][0] for i in range(0, len(predicted_stock_price))]

# close_pred = pd.DataFrame({
#    'Close': data.iloc[1:, 0].values,
#     'Predictions': pred_val
#     }, index = data.index[1:])


# plt.figure(figsize=(10, 6))
# plt.plot(close_pred.index, close_pred['Close'], label='True Values', color='blue')
# plt.plot(close_pred.index, close_pred['Predictions'], label='Predicted Values', color='orange')
# plt.title('True vs Predicted Stock Prices')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.grid(True)
# plt.show()





