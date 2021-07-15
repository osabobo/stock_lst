import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
#from prophet import Prophet
#from prophet.plot import plot_plotly
#from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from plotly import graph_objs as go

from PIL import Image
image = Image.open('index1.jpg')
st.image(image)
st.set_option('deprecation.showfileUploaderEncoding', False)
file_upload = st.file_uploader("Upload csv file for predictions", type="csv")





st.title('Make sure the csv File is in the same format  as stocks.csv before uploading to avoid Error')

if file_upload is not None:
    data = pd.read_csv(file_upload)
    st.subheader("Raw data")
    st.write(data.head())
    st.write(data.tail())

    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock Open'))
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock Close'))

        fig.layout.update(title_text="Time series data and slider under the graph",xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()
    train_dates = pd.to_datetime(data['Date'])
    cols = list(data)[1:6]
    data= data[cols].astype(float)
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    df_for_training_scaled = scaler.transform(data)
    trainX = []
    trainY = []

    n_future = 1   # Number of days we want to predict into the future
    n_past = 14
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:data.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
    trainX, trainY = np.array(trainX), np.array(trainY)
# create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    n_future=90  #Redefining n_future to extend prediction dates beyond original n_future dates...
    forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

    forecast = model.predict(trainX[-n_future:]) #forecast
    forecast_copies = np.repeat(forecast, data.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
    st.subheader("Forcast data")
    st.write(y_pred_future)
