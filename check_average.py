import pickle
import pandas as pd
import numpy as np
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
with open("D:/Users/Polina/test.pickle", 'rb') as f:
    data = pickle.load(f)
average = np.average(data["test_gc"])
anerage_array = np.array([average]*len(data["test_gc"]))
print(anerage_array)
deltas = data["test_gc"] - anerage_array
mape = mean_absolute_percentage_error(data["test_gc"], anerage_array)
print(mape)
print(deltas)