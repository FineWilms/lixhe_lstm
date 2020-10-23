import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import joblib

from keras import optimizers




def Borgharen():
    fp_borgharen =os.path.abspath(os.path.join(os.path.dirname(__file__),'..','rawdata', '2020-10-19_12-25', '6421500_Q_Day.Cmd.txt'))
    df = pd.read_csv(fp_borgharen, skiprows=36, delimiter=';')
    df['date'] = pd.to_datetime(df['YYYY-MM-DD'], format="%Y-%m-%d")
    df.set_index(['date'], inplace=True)
    df.drop(['YYYY-MM-DD','hh:mm'], inplace=True, axis=1)
    df.rename(columns={list(df)[0]: 'discharge_borgharen'}, inplace=True)
    df.replace(-999.000, np.nan, inplace=True)
    return df





def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


df = Borgharen()
start_date = df.index.min()
# end_date = df.index[29219]
end_date = df.index[12000]


# start_date_test = end_date
start_date_test = df.index[29219]

end_date_test = df.index[-1]
mask = (df.index > start_date) & (df.index <= end_date)
df_train = df.loc[mask]

mask_test = (df.index > start_date_test) & (df.index <= end_date_test)
df_test = df.loc[mask_test]



values_test = df_test.values


values = df_train.values
n_steps = 7

data = series_to_supervised(values, n_steps)
data = data.values


data_test = series_to_supervised(values_test, n_steps)
data_test = data_test.values


test_indices = df_test[n_steps::].index

min_max_scaler = preprocessing.MinMaxScaler()




data_scaled = min_max_scaler.fit_transform(data)


joblib.dump(min_max_scaler, 'std_scaler.bin', compress=True)





# X = data_scaled[:,:-1]
# y = data_scaled[:,-1]
"""batchsize is number of samples (each sample has 300 days) to show the model before it updates its weights
if not set the batch size will be equal to the size of all the training samples. Results in memory issues."""
batchsize = 200
val_fraction = 0.3
trainsize = int(data_scaled.shape[0]*(1.0-val_fraction)/batchsize)*batchsize
valsize = int(data_scaled.shape[0]*(val_fraction)/batchsize)*batchsize

print(trainsize, valsize)

X = data_scaled[:trainsize,:-1]
y = data_scaled[:trainsize,-1]

Xval = data_scaled[trainsize:trainsize+valsize,:-1]
yval = data_scaled[trainsize:trainsize+valsize,-1]



scaler = joblib.load('std_scaler.bin')

data_test_scaled = scaler.transform(data_test)


testsize = int(data_test_scaled.shape[0]/batchsize)*batchsize
X_test = data_test_scaled[:testsize,:-1]
y_test = data_test_scaled[:testsize,-1]

"""univariate lstm example"""
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
Xval = Xval.reshape((Xval.shape[0], Xval.shape[1], n_features))

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
print('batchsize is', batchsize)
# define model
model = Sequential()
print(n_steps)
model.add(LSTM(10, activation='relu', batch_input_shape=(batchsize, n_steps, n_features), stateful=True))
model.add(Dense(1))
# optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt = optimizers.Adam(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=opt)
model.compile(optimizer= opt, loss='mse')
"""All the RNN or LSTM models are stateful in theory. These models are meant to remember the entire sequence for
 prediction or classification tasks. However, in practice, you need to create a batch to train a model with 
 backprogation algorithm, and the gradient can't backpropagate between batches. This means that if you have a 
 long time series which does not fit into a single batch, you need to divide the time series into multiple sub-time 
 series and each sub time series goes to separate batch. Then LSTM only remember what happened within a batch. 
 At the initial time point of every batch, states are initialized and set to 0. No previous information. This is very 
 unfortunate because RNN or LSTM are introduced to remember all the past history to predict the next time point. """

# fit model and use early stopping callback to prevent overfitting.
history = model.fit(X, y, epochs=1000, verbose=1, shuffle=False, batch_size=batchsize, validation_data=(Xval, yval), callbacks=[es, mc])

# model.save("model.h5")
print("Saved model to disk")

# # load model
# model = load_model('model.h5')
# # summarize model.
# model.summary()

# # evaluate the model
# score = model.evaluate(X_test, y_test, verbose=1)
# yhat = model.predict(X_test)
#
#
# df_plot = pd.DataFrame()
# df_plot['target'] = y_test
# df_plot['prediction'] = yhat
# df_plot.index= test_indices
# print(df_plot.head())
# df_plot.plot()
# plt.show()
# print(score)

# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
# # plot train and validation loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model train vs validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
# demonstrate prediction
# x_input = array([70, 80, 90])
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
