from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report 
from sklearn.svm import SVC 
import numpy as np
import re

data = np.genfromtxt('hepatitis.csv', skip_header=1, delimiter=',', dtype=object)	# import data
np.random.shuffle(data)	# shuffle data long the 0th axis
data = data[:,1:] # remove first column (the ID column)

processed_data = np.zeros(data.shape, dtype=float)

# convert object array to floats
for r in range(data.shape[0]):
    for c in range(data.shape[1]):
        try:
            processed_data[r,c] = float(data[r,c])
        except(ValueError):
            string_opt = str(data[r,c],'utf-8') # if not int, pick out strings
            if string_opt == '?':
               processed_data[r,c] = np.nan



for c in range(processed_data.shape[1]):    # Anywhere we have nan, we replace w/ col mean
    nan_mask = np.isnan(processed_data[:,c])
    processed_data[nan_mask,c] = np.mean(processed_data[~nan_mask,c])




# Split data
X, y = processed_data[:,1:], processed_data[:,0]

# scale data in each column accordingly
scaleX = OneHotEncoder()
X = scaleX.fit_transform(X)
y = (y-1).astype(int)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.10)


### Default SVC
model = SVC()
model.fit(X_train, y_train) 
model_predictions = model.predict(X_test) 
print(classification_report(y_test, model_predictions)) 
print('\n\n\n')


### Grid search
# create and fir to SVC
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  
          'gamma': ['auto', 0.0001, 0.001, 0.001, 0.01, 0.1, 1, 10], 
          'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'rbf']} 
grid_model = GridSearchCV(SVC(), params, refit = True) 
grid_model.fit(X_train, y_train) 
print(grid_model.best_params_) 
grid_predictions = grid_model.predict(X_test) 
print(classification_report(y_test, grid_predictions)) 




#
