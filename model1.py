import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense
import keras.backend as K
#from sklearn.preprocessing import MhadfddsfinMaxScaler
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import shuffle


def measure_accuracy(y_true, y_pred):
    return K.mean(K.abs(y_true-y_pred))


#scaler = MinMaxScaler(feature_range=(0,1))

features = pd.read_csv('./data/train_features.csv')
features = features.drop(['week_start_date'],axis=1)

labels = pd.read_csv('./data/train_labels.csv')

# combine features and labels
merged = pd.merge(features,labels)
merged = shuffle(merged)
# merged = merged.loc[labels['city'] == 'sj']

features = merged.drop(['city','year','weekofyear','total_cases'],axis=1).values
features = np.nan_to_num(features)
labels = merged[['total_cases']].values

X_train = features[:]
y_train = labels[:]

X_test = features[-100:]
y_test = labels[-100:]

regr = KernelRidge(alpha=1)
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print("Mean squared error: %.2f"
      % mean_absolute_error(y_test, y_pred))


test_original = pd.read_csv('./data/dengue_features_test.csv')
test = test_original.copy()
test_features = test.drop(['week_start_date','city','year','weekofyear'],axis=1).values
test_features = np.nan_to_num(test_features)

test_original_write = test_original.values
test_pred = regr.predict(test_features)

answer = []
for i in range(test_original_write.shape[0]):
    record = test_original_write[i]
    result = [record[0], record[1], record[2], int(test_pred[i][0])]
    answer.append(result)
    print(result)

a = np.asarray(answer)
df = pd.DataFrame(a)
df.to_csv('results.csv')