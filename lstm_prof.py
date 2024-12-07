### This module uses Stacked LSTM 50 units and 100 epochs to predict Google stock prices from Yahoo Finance, epochs = 100
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import holidays
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam

csv_directory = "historical_data" 

data_frames = {}

for file in os.listdir(csv_directory):
    if file.endswith(".csv"): 
        asset_name = file.split(".csv")[0]
        file_path = os.path.join(csv_directory, file)
        
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        
        # CHANGE HERE IF U DONT WANT TO USE ADJ CLOSE
        data_frames[asset_name] = df["Adj Close"].rename(asset_name)

combined_df = pd.concat(data_frames.values(), axis=1)
combined_df = combined_df.reset_index()
#Taking 2018 as start date to account for crypto
df_filtered = combined_df[combined_df['Date'] >= '2021-01-01']
#Taking a subset of the stocks
filtered_df = df_filtered.loc[:, ['Date', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
       'XLRE', 'XLU', 'XLV', 'XLY', 'ADA-USD', 'BTC-USD', 'XRP-USD']]
#print(filtered_df)
#print(filtered_df.iloc[1572:1669,:])

#Filter out days that are NOT trading days and are Holidays
us_holidays = holidays.US()
filtered_df['Holiday'] = filtered_df.Date.apply(lambda x: x in us_holidays)
filtered_df['Day'] = filtered_df['Date'].dt.day_name()
filtered_df = filtered_df.loc[filtered_df['Holiday'] == False]
filtered_df = filtered_df.loc[~filtered_df.Day.isin(['Saturday', 'Sunday'])]

print(filtered_df.isnull().sum())
#see sample of what's null
#print(filtered_df[filtered_df.isnull().any(axis=1)].sample(5))

'''
#calculate the percentage change row wise
df_pct_change = filtered_df.iloc[:, 1:-2].pct_change()
# Add the Date column back to the percentage change DataFrame
df_pct_change['Date'] = filtered_df['Date']
# Rearrange the columns to put Date first
df_pct_change = df_pct_change[['Date'] + [col for col in df_pct_change.columns if col != 'Date']]
print(df_pct_change)
'''
#print(filtered_df)

stocks = filtered_df.columns[1:14]  # Exclude the 'Date' column

# Initialize dictionaries to store results for all stocks
pred_prices = {}
act_prices = {}


for stock in stocks:
    print (f"Processing stock: {stock}")
    p=filtered_df['XLY']
    p=filtered_df[stock]
    #print(p)


    split_index = int(len(p) * 0.87)
    # Split the data
    training_set = p.iloc[:split_index]  # First 87% for training
    test_set = p.iloc[split_index:]     # Remaining 13% for testing
    # Drop NaN values
    training_set.dropna(inplace=True)
    test_set.dropna(inplace=True)
    training_set = training_set.values
    trgsset=pd.DataFrame(training_set)
    test_set = test_set.values
    tstset=pd.DataFrame(test_set)


    '''
    training_set=p.iloc[:1512] ### keeps the Adj Close prices
    training_set.dropna(inplace=True)

    test_set=p.iloc[1512:]
    test_set.dropna(inplace=True)
    #print('TEST DATA')
    #print(tstset)
    '''

    print(p.shape, trgsset.shape, tstset.shape)
    print(trgsset.head(),trgsset.tail(), tstset.head(),tstset.tail())

    ### Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc=MinMaxScaler(feature_range=(0,1))
    training_set = training_set.reshape(-1,1)
    training_set_scaled=sc.fit_transform(training_set)
    test_set = test_set.reshape(-1,1)
    test_set_scaled=sc.transform(test_set)
    type(training_set_scaled), type(test_set_scaled), 
    #assert not np.isnan(training_set_scaled).any(), "NaN in training data"
    #assert not np.isnan(test_set_scaled).any(), "NaN in test data"

    ### creating data structure with 60 time-steps and 1 output
    X_train=[]
    y_train=[]
    for i in range(60,len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)   
    print(X_train.shape, y_train.shape)
    X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1)) 
    ### this step converts X_train to 3D from (1738,60) to (1738,60,1) for input to the keras app

    print("NaNs in X_train:", np.isnan(X_train).any())
    print("NaNs in y_train:", np.isnan(y_train).any())

    ### Initializing RNN
    model = Sequential()

    ### Add first LSTM layer and add Dropout Reegularization
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1))) ### Sequential reads input as 3D
    model.add(Dropout(0.2))
    ### add return_sequences=True for all LSTM layers except the last one. Setting this flag to True lets Keras know that 
    ### LSTM output should contain all historical generated outputs along with time stamps (3D). So, next LSTM layer can work 
    ### further on the data. If this flag is false, then LSTM only returns last output (2D). Such output is not good enough 
    ### for another LSTM layer.

    ### Add second LSTM layer and Dropout
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    ### Add third LSTM layer and Dropout
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    ### Add fourth LSTM layer and Dropout
    model.add(LSTM(units=50)) ### note: last LSTM layer does not carry argument 'return_sequences=True'
    model.add(Dropout(0.2))
    ### Add output layer
    model.add(Dense(units=1))  ### not capital "U"nit
    ### Compiling the RNN
    #optimizer = Adam(clipvalue=1.0)
    #model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.compile(optimizer='adam',loss='mean_squared_error')

    ### Run the training set with the LSTM (specialized RNN here)
    model.fit(X_train,y_train,epochs=20,batch_size=10)

    model.summary()

    predict_train=model.predict(X_train) 
    print(predict_train.shape) 
    ### this output is (1040,60,1), we want only the first no. in each row of 1040 
    ### this 3D structure makes it more difficult to interpret comparison of prediction in training set vs output in trg set

    predict_train=predict_train[:, 0]  ### select first column or first day of 60 days of each day prediction from 1 to 1040

fraction = 0.2 
start_index = int(len(test_set_scaled) * (1 - fraction))
#start_index = len(test_set_scaled) - 165

### creating data structure with 60 time-steps and 1 output
X_test=[]
y_test=[]  ### here y_test is to collect the predicted y values


time_steps = 60
#start_index (replaced)
for i in range(time_steps,len(test_set_scaled)):
    X_test.append(test_set_scaled[i-time_steps:i, 0])
    y_test.append(test_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)  ### convert to np arrays as X_test is "list"    
X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

y_test = np.reshape(y_test, (-1, 1))
y_train=np.reshape(y_train,(-1,1))

### Prediction
predicted_stock_price=model.predict(X_test)
predicted_stock_price=predicted_stock_price[:,0]
pred_prices[stock] = predicted_stock_price
#predicted_stock_price1=sc.inverse_transform(predicted_stock_price)
#predicted_stock_price.shape



from sklearn.metrics import mean_squared_error
import math
def print_error(trainY, testY, train_predict, test_predict):    
    ### Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    ### Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))
    
print_error(y_train, y_test, predict_train, predicted_stock_price)

print(predicted_stock_price)
Pred_price=sc.inverse_transform(predicted_stock_price.reshape(-1, 1))
#Pred_price=Pred.price.reshape(-1, 1)

actual_prices = test_set[time_steps:, 0]
#ensure lengths match, trim to match predictions
actual_prices = actual_prices[:Pred_price.shape[0]] 

print(f"Actual Prices Shape: {actual_prices.shape}, Predicted Prices Shape: {Pred_price.shape}")
### Plotting the stock price prediction results
plt.plot(actual_prices,label="Actual Price",linestyle='--') ### from observation 60 to 226 till 28 Nov 2024
plt.plot(Pred_price,label="Predicted Price",linestyle='-')
plt.title('Stock Price Prediction till 28 Nov 2024')
plt.xlabel('Time')
plt.ylabel('Price in $')
plt.legend()
plt.show()
