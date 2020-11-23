from sklearn.model_selection import train_test_split as tts
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
import numpy as np
import re
import sys

# Import and process data
data = np.genfromtxt('BackOrders.csv',delimiter=',', skip_header=1, dtype=object)
np.random.shuffle(data) # shuffle data along the 0th axis
data = data[:,1:] # get rid of sku

processed_data = np.zeros(data.shape, dtype=float)

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
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.10)


# plotting method
def plot_model(history, number, name):	# model, model number
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model ' + str(number) + ' loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig(name + '_loss.png')

	plt.figure()
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model ' + str(number) + ' accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig(name + '_accuracy.png')

# model 1
# Test Accuracy -->  
model1 = Sequential()
model1.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history1 = model1.fit(X_train, y_train, validation_split=0.3, epochs=100)
_, accuracy = model1.evaluate(X_test, y_test)
print(model1.summary())
print('\033[1m\033[92m Accuracy: %.2f \033[0m' % (accuracy*100))
plot_model(history1, 1, '2a')


# model 2, 100 epochs
# Test Accuracy -->  
model2 = Sequential()
model2.add(Dense(15, input_dim=X_train.shape[1], activation='tanh'))
model2.add(Dense(1, input_dim=15, activation='tanh'))
model2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, validation_split=0.3, epochs=100)
_, accuracy = model2.evaluate(X_test, y_test)
print(model2.summary())
print('\033[1m\033[92m Accuracy: %.2f \033[0m' % (accuracy*100))
plot_model(history2, 2, '2b')


# model 3, 100 epochs 
# Test Accuracy -->  
model3 = Sequential()
model3.add(Dense(25, input_dim=X_train.shape[1], activation='tanh'))
model3.add(Dense(15, input_dim=25, activation='tanh'))
model3.add(Dense(1, input_dim=15, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history3 = model3.fit(X_train, y_train, validation_split=0.3, epochs=100)
_, accuracy = model3.evaluate(X_test, y_test)
print(model3.summary())
print('\033[1m\033[92m Accuracy: %.2f \033[0m' % (accuracy*100))
plot_model(history3, 3, '2c')



# model 4, 100 epochs 
# Test Accuracy -->  
model4 = Sequential()
model4.add(Dense(25, input_dim=X_train.shape[1], kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='tanh'))
model4.add(Dense(15, input_dim=25, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='tanh'))
model4.add(Dense(1, input_dim=15, activation='sigmoid'))
model4.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history4 = model4.fit(X_train, y_train, validation_split=0.3, epochs=100)
_, accuracy = model4.evaluate(X_test, y_test)
print(model4.summary())
print('\033[1m\033[92m Accuracy: %.2f \033[0m' % (accuracy*100))
plot_model(history4, 4, '2d')






#
