# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 22:58:31 2018

@author: kennedy
"""

from datetime import datetime
from PredictiveModel import Model
from DataCollector import FetchData, NormalizeData, Predictors
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataframe = FetchData("F", "yahoo", datetime(2000, 1, 1), datetime.now()).fetch()
    pred = Predictors(dataframe)
    Xf, Yf, X_train, X_test, Y_train, Y_test = NormalizeData(dataframe, 7, 15).normalizeData()
    split = int(0.7*len(dataframe))
    GSCV = Model().optimize(X_train, X_test, Y_train, Y_test)
    dataframe['Predicted_Signal'] = GSCV.predict(Xf)
    dataframe['Stock_returns'] = np.log(dataframe['Close']/dataframe['Close'].shift(1))
    Cumulative_returns = np.cumsum(dataframe[split:]['Stock_returns'])
    dataframe['Startegy_returns'] = dataframe['Stock_returns']* dataframe['Predicted_Signal'].shift(1)
    Cumulative_Strategy_returns = np.cumsum(dataframe[split:]['Startegy_returns'])
    plt.figure(figsize=(10,5))
    plt.plot(Cumulative_returns, color='r',label = ' Stock_returns')
    plt.plot(Cumulative_Strategy_returns, color='g', label = 'Strategy Returns')
    plt.legend()
    plt.show()
        
    
    
