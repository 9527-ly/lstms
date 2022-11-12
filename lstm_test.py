# -*- coding: gbk -*-
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Dropout, GlobalAvgPool2D, GlobalAveragePooling2D,MaxPooling2D, BatchNormalization, Conv2D,LSTM
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam, Adadelta
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import mean_absolute_error, mean_absolute_percentage_error
import math
import scipy.io as scio


np.random.seed(12345)
tf.random.set_seed(1234)

#Data set
def LSTMs(mode):
    if mode == 'lstm':
        data = scio.loadmat('ji1.mat')
        x=data['xin1']
        x=x.reshape(-1,127,1)
        x=x[:2000,:,:]
        y = np.load('y_input.npy', allow_pickle=True)
        y=y[:2000,0]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    else:
        mode == 'Convlstm'
        x, y = np.load('x_input.npy', allow_pickle=True), np.load('y_input.npy', allow_pickle=True)
        x = x.reshape(-1, 1, 10, 10, 6)
        x = x[:2000, :, : ,: ,: ]
        y = y[:2000, 0]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#Select Model
mode = 'lstm'
# mode = 'Convlstm'
X_train, X_test, y_train, y_test = LSTMs(mode)

#model
def LSTM_Model(input_shape=(X_train.shape[1], 1)):
    model = Sequential()
    model.add(LSTM(32,activation='relu',input_shape=input_shape,return_sequences=True))
    model.add(LSTM(64,activation='relu', return_sequences=True))
    model.add(LSTM(64,activation='relu', return_sequences=False))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(0.0001), loss='mae')
    print(model.summary())
    return model

def Conv_lstm(input_shape=(X_train.shape[1], 10, 10, 6)):
    model = Sequential()
    model.add(ConvLSTM2D(filters=16, kernel_size=(3,3) ,padding='SAME',activation='relu',input_shape=input_shape,return_sequences=True))#
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),padding='SAME', return_sequences=False))
    model.add(MaxPooling2D(pool_size=[4, 4]))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(0.0001), loss='mae')
    print(model.summary())
    return model



if mode == 'lstm':
    model = LSTM_Model()
else:
    model = Conv_lstm()

# Weight before model training
initial_weight=model.get_weights()

# model training
reduce = ReduceLROnPlateau(monitor='loss',factor=0.1,patience=20)
model.fit(X_train, y_train, epochs=100, callbacks=[reduce], shuffle=True, batch_size=16)

# Weight after model training
trained_weight=model.get_weights()

# Test error
predicted = model.predict(X_test)
error = np.abs(predicted - y_test)

#model save
if mode == 'lstm':
    model.save('lstm.h5')
else:
    model.save('convlstm.h5')
