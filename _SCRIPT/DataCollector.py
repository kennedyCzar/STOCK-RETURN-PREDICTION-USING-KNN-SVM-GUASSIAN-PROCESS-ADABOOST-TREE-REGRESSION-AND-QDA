# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 00:44:16 2018

@author: kennedy
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from TechnicalIndicators import TechnicalIndicators
from datetime import datetime



class FetchData(object):
    
    def __init__(self, data, source, start, end):
        '''
        Arguments:
            @data: takes in the name of the stock
                    for analysis
            @source: API from where data is fetched from
                    mostly yahoo.
            @start: Start date for data source in the format
                    e.g 
                    >>datetime(2000, 1, 1)
            @end: the end of the timeseries for data collection
                    e.g 
                    >>datetime.now()
            @train: This is boolean for checking train or 
                    test score.
    
        Use Case:
            start_date = datetime(2000, 1, 1)
            end_date = datetime.now()
            dataframe = web.DataReader("TSLA", "yahoo", start_date, end_date)
            dataframe = fetchData("TSLA", "yahoo", start_date, end_date).fetch()
        '''
        self.data = data
        self.source = source
        self.start = start
        self.end = end
        self.train = False
        
    def fetch(self):
        if not self.data:
            raise ValueError('Stock name not available')
        elif not self.start or not self.end:
            raise ValueError('Start date and End date not specified')
        else:
            dataframe = web.DataReader(self.data, self.source, self.start, self.end)
        return dataframe
       
dataframe = FetchData("TSLA", "yahoo", datetime(2000, 1, 1), datetime.now()).fetch()

class Predictors:
    def __init__(self, data):
        '''
        Contains a list of technical indicators.
        This indicators are what we put together
        For this project we would focus on the  following Indicators
            --Exponential Moving Average --> For predicting Trend
            --Commodity Channel Index --> Forecasting the future price move
            --Momentum --> The rate of change of price especially clossing
            --Stochastic Oscillator --> Future price movement
            --Mass Index -->Average Exponential Index
            --Force -->Relative Strength Index
        '''
        self.dataframe = data
        return self.predictors()
    
    def predictors(self):
        self.dataframe['MA'] = TechnicalIndicators.moving_average(self.dataframe, 10)
        self.dataframe['EMA'] = TechnicalIndicators.exponential_moving_average(self.dataframe, 10)
        self.dataframe['CCI'] = TechnicalIndicators.commodity_channel_index(self.dataframe, 10)
        self.dataframe['MMT'] = TechnicalIndicators.momentum(self.dataframe, 10)
        self.dataframe['SOD'] = TechnicalIndicators.stochastic_oscillator_d(self.dataframe, 10)
        self.dataframe['SOK'] = TechnicalIndicators.stochastic_oscillator_k(self.dataframe)
        self.dataframe['FI'] = TechnicalIndicators.force_index(self.dataframe, 10)
        self.dataframe['MI'] = TechnicalIndicators.mass_index(self.dataframe, 10)
        return self.dataframe.fillna(0, inplace=True)
        
        
#pred = Predictors(dataframe)  

class NormalizeData(object):
    def __init__(self,data, short_price, long_price):
        
        '''
        :short_price: 
                    MA short and Long
        :long_price:
                    MA Long and Short
        '''
        self.short_price = short_price
        self.long_price = long_price
        self.data = data
        
    def normalizeData(self):
        Xf = pd.DataFrame(MinMaxScaler().fit_transform(self.data.drop(['Close'], axis = 1).values))
        Xf.columns = dataframe.columns.drop(['Close'])
        if not self.long_price > self.short_price:
            raise ValueError('Short price should be less than long')
        elif self.short_price == self.long_price:
            raise ValueError('Prices should not be the same')
        else:
            '''
            The Yf states:
                When short_price is greater than Long_price--> Buy
                When short_price is less than long_price --> Sell
            '''
#            Yf = np.where(self.data['Close'].rolling(window = self.short_price).mean() > self.data['Close'].rolling(window = self.long_price).mean(),1,0)
            Yf = np.where(self.data['Close'].shift(-1) > self.data['Close'],1,0)
        #Train/Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(Xf, Yf, test_size = 0.3, random_state = False)
        return Xf, Yf, X_train, X_test, Y_train, Y_test
        


class feature_importance(object):
    def __init__(self, Xf, Yf):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=250, max_depth=25)
        self.model.fit(Xf, Yf)
        importance = pd.DataFrame({'features': Xf.columns,
                        'importances': self.model.feature_importances_})
        importance = importance.sort_values('importances', ascending=False)
        plt.figure(figsize = (16,12))
        sns.barplot(importance.importances, importance.features)
        plt.title('Feature Importance Plot')
        plt.show()
        
#feature_importance(dataframe, Yf)
#Xf, Yf, X_train, X_test, Y_train, Y_test = NormalizeData(7, 15).normalizeData()