import datetime
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from keras.layers import LSTM, Activation, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.losses import MSE
from keras.optimizers import Adam
from keras.activations import relu



class StockPrediction():

    def LSTM_Model(self, X_train, y_train):
        LSTM_Model = Sequential()

        LSTM_Model.add(LSTM(units=96, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        LSTM_Model.add(Dropout(0.2))
        LSTM_Model.add(LSTM(units=96, return_sequences=True))
        LSTM_Model.add(Dropout(0.2))
        LSTM_Model.add(LSTM(units=96, return_sequences=True))
        LSTM_Model.add(Dropout(0.2))
        LSTM_Model.add(LSTM(units=96))
        LSTM_Model.add(Dropout(0.2))
        LSTM_Model.add(Dense(units=1))

        LSTM_Model.compile(loss=MSE, optimizer=Adam())
        LSTM_Model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_split=0.5)
        return LSTM_Model


    def create_dataset(self, df, n, feature=0):
        x = []
        y = []

        for i in range(n, df.shape[0]):
            x.append(df[i-n:i, feature])
            y.append(df[i, feature])
        x = np.expand_dims(np.array(x), -1)
        y = np.expand_dims(np.array(y), -1)
        return x, y

    def evaluate_model(self, model, test_data, target_data):
        y_pred_test = model.predict(test_data)

        try:
            y_pred_test = y_pred_test.yhat
        except:
            pass

        test_rs = r2_score(target_data, y_pred_test)
        print('R Squared : ', round(test_rs, 5))

        test_MAE = mean_absolute_error(target_data, y_pred_test)
        print("Mean Absolute Error : ", round(test_MAE, 5))

        plt.figure(figsize=(20,10))
        plt.plot(y_pred_test, color='green', marker='o', linestyle='dashed',label='Predicted Price')
        plt.plot(target_data, color='red', label='Actual Price')
        plt.title('Comparison of actual and predicted stock prices for ' + model_name)
        plt.xlabel('Day')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()

        return test_rs, test_MAE

    def train(self, ticker, startDate):
        TODAY = datetime.date.today()
        data = yf.download(ticker, startDate, TODAY.strftime("%Y-%m-%d"))
        data.head()
        data["Adj Close"].plot(title=f"{ticker} Stock Adjusted Closing Price.")

        # data preparation
        split = int(data.shape[0] * 0.8)
        df_train = data[:split]
        df_test = data[split:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_train = scaler.fit_transform(df_train)
        
        dataset_test = scaler.transform(df_test)

        X_train, y_train = self.create_dataset(dataset_train, 50, 6)
        X_test, y_test = self.create_dataset(dataset_test, 50, 6)

        


        



    def predict(ticker, days):
        pass

    def convert(prediction_list):
        pass