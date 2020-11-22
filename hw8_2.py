from sklearn.model_selection import train_test_split as tts
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import re
import sys

# Import and process data
data = np.genfromtxt('BackOrders.csv',delimiter=',', skip_header=1, dtype=object)
data = data[:,1:] # get rid of sku

processed_data = np.zeros(data.shape, dtype=float)

DEBUG = False

# convert object array to floats
for r in range(data.shape[0]):
    for c in range(data.shape[1]):
        try:
            processed_data[r,c] = float(data[r,c])
        except(ValueError):
            string_opt = str(data[r,c],'utf-8') # if not int, pick out strings
            string_opt = re.sub(r'[^\w\s]','',string_opt)   # remove punctuation
            if string_opt == 'NA':
                processed_data[r,c] = np.nan
            elif string_opt == 'No':        # Only cols 11, 16: are categorical w/ No, Yes
                processed_data[r,c] = 0
            elif string_opt == 'Yes':
                processed_data[r,c] = 1

for c in range(processed_data.shape[1]):    # Anywhere we have nan, we replace w/ col mean
    nan_mask = np.isnan(processed_data[:,c])
    processed_data[nan_mask,c] = np.mean(processed_data[~nan_mask,c])

X, y = processed_data[:,:-1], processed_data[:,-1] # divide into the X and y datasets

# scale data in each column accordingly
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# shuffle and split
y = y[:, np.newaxis]
scaled_data = np.hstack([y, X_scaled])
np.random.shuffle(scaled_data) # shuffle data along the 0th axis
y, X = scaled_data[:,0], scaled_data[:,1:]
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.10, random_state=42)

# # model 1
# model1 = Sequential()
# model1.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
# model1.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# history1 = model1.fit(X_train, y_train, validation_split=0.1, epochs=100)
# _, accuracy = model1.evaluate(X_test, y_test)
# print('Accuracy: %.2f' % (accuracy*100))
#
# # summarize history for loss for model 1
# plt.figure()
# plt.plot(history1.history['loss'])
# plt.figure()
# plt.plot(history1.history['val_loss'])
# plt.title('Model 1 loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')


# model 2
model2 = Sequential()
model2.add(Dense(15, input_dim=X_train.shape[1], activation='tanh'))
model2.add(Dense(1, input_dim=15, activation='tanh'))
model2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, validation_split=0.1, epochs=100)
_, accuracy = model2.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# summarize history for loss for model 1
plt.figure()
plt.plot(history2.history['loss'])
plt.figure()
plt.plot(history2.history['val_loss'])
plt.title('Model 2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')





plt.show()

#
