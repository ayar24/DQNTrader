
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, GRU
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import yfinance as yf
import time
########################################################################################################################
# INPUTS:
symbols = ['JPM']
########################################################################################################################
#####################################################################################################################################################################################################
#lstm
forward_prediction_period = 1 # how many forward days one wants to predict price, I would leave this alone at 1
lookback = 3 # look back number of days

for symbol in symbols:

    # read in datad
    del_time = "Datetime"
   # del_time = "Date"

    start_date = pd.Timestamp.today().floor('D') - pd.Timedelta(days=3)
    end_date = pd.Timestamp.today()
    df = yf.download(symbol, start=start_date, end=end_date, interval='5m')

    #df = yf.download(symbol, period="1y", interval="1d")
    print(df)
    #time.sleep(30000)

    # LSTM MODEL CODE LOGIC
    data_open = df['Open']
    data_open = pd.DataFrame(data_open)
    data_open = data_open.reset_index()

    del data_open[del_time]
    print(data_open)

    data_high = df['High']
    data_high = pd.DataFrame(data_high)
    data_high = data_high.reset_index()

    del data_high[del_time]
    print(data_high)

    data_low = df['Low']
    data_low = pd.DataFrame(data_low)
    data_low = data_low.reset_index()
    del data_low[del_time]
    print(data_low)

    data_close = df['Close']
    data_close = pd.DataFrame(data_close)
    data_close = data_close.reset_index()
    del data_close[del_time]
    print(data_close)

    data_volume = df['Volume']
    data_volume = pd.DataFrame(data_volume)
    data_volume = data_volume.reset_index()

    del data_volume[del_time]
    print(data_volume)

    data_sma_20 = df['Close'].rolling(window=20).mean()
    data_sma_20 = pd.DataFrame(data_sma_20)
    data_sma_20 = data_sma_20.reset_index()

    del data_sma_20[del_time]
    data_sma_20 = data_sma_20.fillna(0)
    data_sma_20.columns = ['sma20']
    print(data_sma_20)

    data_sma_50 = df['Close'].rolling(window=50).mean()
    data_sma_50 = pd.DataFrame(data_sma_50)
    data_sma_50 = data_sma_50.reset_index()

    del data_sma_50[del_time]
    data_sma_50 = data_sma_50.fillna(0)
    data_sma_50.columns = ['sma50']
    print(data_sma_50)

    data_sma_200 = df['Close'].rolling(window=200).mean()
    data_sma_200 = pd.DataFrame(data_sma_200)
    data_sma_200 = data_sma_200.reset_index()

    del data_sma_200[del_time]
    data_sma_200 = data_sma_200.fillna(0)
    data_sma_200.columns = ['sma200']
    print(data_sma_200)

 #########################################################################
    # Calculate - RSI , CCI, MCAD and ADX

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data_rsi = 100 - (100 / (1 + rs))
    data_rsi = pd.DataFrame(data_rsi)
    data_rsi = data_rsi.reset_index()

    del data_rsi[del_time]
    data_rsi = data_rsi.fillna(0)
    data_rsi.columns = ['RSI']
    print(data_rsi)

    # Calculate - CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=20).mean()
    md = (tp - ma).abs().rolling(window=20).mean()
    data_cci = (tp - ma) / (0.015 * md)
    data_cci = pd.DataFrame(data_cci)
    data_cci = data_cci.reset_index()

    del data_cci[del_time]
    data_cci = data_cci.fillna(0)
    data_cci.columns = ['CCI']
    print(data_cci)

    # Calculate - ADX
    up = df['High'].diff().clip(lower=0)
    down = -df['Low'].diff().clip(upper=0)
    tr = pd.concat([up, down], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    pos_dm = ((df['High'] - df['High'].shift(1)).clip(lower=0))
    neg_dm = ((df['Low'].shift(1) - df['Low']).clip(lower=0))
    pos_di = 100 * pos_dm.rolling(window=14).sum() / atr
    neg_di = 100 * neg_dm.rolling(window=14).sum() / atr
    data_adx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di)).rolling(window=14).mean()
    data_adx = pd.DataFrame(data_adx)
    data_adx = data_adx.reset_index()

    del data_adx[del_time]
    data_adx = data_adx.fillna(0)
    data_adx.columns = ['ADX']
    print(data_adx)

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    data_mcad = macd_line - signal_line
    data_mcad = pd.DataFrame(data_mcad)
    data_mcad = data_mcad.reset_index()

    del data_mcad[del_time]
    data_mcad = data_mcad.fillna(0)
    data_mcad.columns = ['MCAD']
    print(data_mcad)

    data = pd.concat([data_open, data_high, data_low, data_close, data_volume, data_sma_20, data_sma_50, data_sma_200, data_rsi, data_cci, data_mcad, data_adx], axis=1)
    data = pd.DataFrame(data)
    data = data.reset_index()
    del data['index']
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'sma20', 'sma50', 'sma200','RSI','CCI','MCAD','ADX']
    print(data)

    #########################################################################
    # Convert default of 1 day bars to n period bars for look forward prediction period:
    data = data[data.index % forward_prediction_period == 0]
    print("DATA TEST")
    print(data)

    ########################################################################

    sc1 = MinMaxScaler(feature_range=(0, 1))
    sc2 = MinMaxScaler(feature_range=(0, 1))
    sc3 = MinMaxScaler(feature_range=(0, 1))
    sc4 = MinMaxScaler(feature_range=(0, 1))
    sc5 = MinMaxScaler(feature_range=(0, 1))
    sc6 = MinMaxScaler(feature_range=(0, 1))
    sc7 = MinMaxScaler(feature_range=(0, 1))
    sc8 = MinMaxScaler(feature_range=(0, 1))
    sc9 = MinMaxScaler(feature_range=(0, 1))
    sc10 = MinMaxScaler(feature_range=(0, 1))
    sc11 = MinMaxScaler(feature_range=(0, 1))
    sc12 = MinMaxScaler(feature_range=(0, 1))
    input_feature = data[['Open', 'High', 'Low', 'Close', 'Volume', 'sma20', 'sma50', 'sma200','RSI','CCI','MCAD','ADX']].values

    symbol_open = input_feature[:, 0]
    symbol_high = input_feature[:, 1]
    symbol_low = input_feature[:, 2]
    symbol_close = input_feature[:, 3]
    symbol_volume = input_feature[:, 4]
    symbol_sma20 = input_feature[:, 5]
    symbol_sma50 = input_feature[:, 6]
    symbol_sma200 = input_feature[:, 7]
    symbol_rsi = input_feature[:, 8]
    symbol_cci = input_feature[:, 9]
    symbol_mcad = input_feature[:, 10]
    symbol_adx = input_feature[:, 11]

    input_data_1 = sc1.fit_transform(symbol_open.reshape(-1, 1))
    input_data_2 = sc2.fit_transform(symbol_high.reshape(-1, 1))
    input_data_3 = sc3.fit_transform(symbol_low.reshape(-1, 1))
    input_data_4 = sc4.fit_transform(symbol_close.reshape(-1, 1))
    input_data_5 = sc5.fit_transform(symbol_volume.reshape(-1, 1))
    input_data_6 = sc6.fit_transform(symbol_sma20.reshape(-1, 1))
    input_data_7 = sc7.fit_transform(symbol_sma50.reshape(-1, 1))
    input_data_8 = sc8.fit_transform(symbol_sma200.reshape(-1, 1))
    input_data_9 = sc9.fit_transform(symbol_rsi.reshape(-1, 1))
    input_data_10 = sc10.fit_transform(symbol_cci.reshape(-1, 1))
    input_data_11 = sc11.fit_transform(symbol_mcad.reshape(-1, 1))
    input_data_12 = sc12.fit_transform(symbol_adx.reshape(-1, 1))

    input_data = np.hstack((input_data_1, input_data_2, input_data_3, input_data_4,input_data_5, input_data_6, input_data_7, input_data_8, input_data_9,input_data_10,input_data_11,input_data_12))

    test_size = int(.3 * len(data))
    X = []
    y = []
    for i in range(len(data) - lookback - 1):
        t = []
        for j in range(0, lookback):
            t.append(input_data[[(i + j)], :])
        X.append(t)
        y.append(input_data[i + lookback, 1])

    X, y = np.array(X), np.array(y)
    X_test = X[test_size + lookback:]
    Y_test = y[test_size + lookback:]
    X = X.reshape(X.shape[0], lookback, 12)
    X_test = X_test.reshape(X_test.shape[0], lookback, 12)
    print(X.shape)
    print(X_test.shape)

    # BUILD THE RNN MODEL

    # We add 30 RNN cells that will be stacked one after the other in the RNN, implementing an efficient stacked RNN.
    # return_sequencesis True to return the last output in the output sequence.
    # input_shape will be of the 3D format of test sample size, time steps, no. of input features. output one unit.

    model = Sequential()
    model.add(Dropout(0.2, input_shape=(X.shape[1], 12)))  # can adjust dropout %
    model.add(Activation('relu'))  # can likely replace and test with sigmoid or softmax also to see if results improve
    model.add(LSTM(units=30, return_sequences=True, input_shape=(X.shape[1], 12)))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(LSTM(units=30))
    model.add(Dense(units=1))
    model.summary()

    # now compile the model using adam optimizer and loss function will be mean squared error for the regression problem
    model.compile(optimizer='adam', loss='mean_squared_error')

    # now fit the data to the input data using batch_size of 32 and 100 epochs
    model.fit(X, y, epochs=10, batch_size=32, validation_data=(X_test, Y_test),validation_split=0.1)  # change values to test results, default at 10% testing data (0.1)

    #  finally predict the symbol prices of the test data
    predicted_value = model.predict(X_test)


    predicted_symbol_price = sc2.inverse_transform(predicted_value)
    predicted_symbol_price = pd.DataFrame(predicted_symbol_price)
    #print(data['Close'],predicted_symbol_price)
    print (predicted_symbol_price)
    lstm_predicted_symbol_value = predicted_symbol_price.values[-1:]

    empty_df = pd.DataFrame(columns=['Values'])
    n = data.shape[0]-predicted_symbol_price.shape[0]; # Number of empty rows to add
    for i in range(n):
       empty_df.loc[i] = lstm_predicted_symbol_value[0]
    #empty_df = pd.DataFrame({'Values': []}, index=range(n))
    new_df = pd.concat([empty_df, predicted_symbol_price]).reset_index(drop=True)

    plt.plot(data['Close'])
    plt.plot(new_df)

# Add labels and title
   # plt.xlabel('X')
  #  plt.ylabel('Y')
   # plt.title('Plot of DataFrame Values')

# Display the plot
    plt.show()

    print("PREDICTED VALUE FOR:")
    print(symbol)
    print("AT A PERIOD OF:")
    print(forward_prediction_period)
    print("DAY(S) IS.........")


    print(lstm_predicted_symbol_value)