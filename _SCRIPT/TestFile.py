# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:22:58 2018

@author: kennedy
property of
"""

import pandas as pd
import numpy as np
import graphviz 
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from TechnicalIndicators import TechnicalIndicators
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate



#%% CONSTANTS


N_NEIGBORS = 10
KERNELS = ['linear', 'rbf']
GAMMA = [0.0001, 0.001, 0.01, 1]
CRITERION = ['gini', 'entropy']
MAX_DEPTH = 5
MAX_FEATURES = ['auto', 'sqrt', 'log2']
N_VALIDATION = 2
N_COMPONENTS = 2
BEST_ACCURACY = 0.0
BEST_CLASSIFIER = 0
BEST_GRIDSEARCH = ''


#%% LOAD DATA
#data.Close.plot() 
Xf = pd.DataFrame(MinMaxScaler().fit_transform(data.drop(['Close', 'Volume'], axis = 1).values))

Yf = np.where(data['Close'].shift(-1) > data['Close'],1,0)
#Yf = pd.DataFrame(np.array(data.loc[:, ['Close']]), columns = ['Close'])
X_train, X_test, Y_train, Y_test = train_test_split(Xf, Yf, test_size = 0.3, random_state = False)

#k_fold = KFold(n_splits = 3, random_state=None, shuffle=False)
#for train, test in k_fold.split(Xf):
#    X_train, X_test = Xf[train], Xf[test]
#    Y_train, Y_test = Yf[train], Yf[test]

#classifiers to train our dataset
classifiers = {'KNN': KNeighborsClassifier(3), 'SVC': SVC(kernel="linear", C=0.025), 'SVC2': SVC(gamma=2, C=1),
              'GuassianClassifier': GaussianProcessClassifier(1.0 * RBF(1.0)), 'LinearRegression': LinearRegression(), 
              'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=5), 
              'RandomForest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), 'Adaboost': AdaBoostClassifier(),
              'GuassianNB': GaussianNB(), 'QuadraticDiscriminant': QuadraticDiscriminantAnalysis()}


#model = KNeighborsClassifier(n_neighbors = 5)
#model.fit(X_train, Y_train)
#prediction = model.predict(X_test)
#
#plt.plot(Xf)
#
#plt.plot(X_test, prediction)
#
#
#accuracy_score(Y_test, prediction)
#confusion_matrix(Y_test, prediction)
#print(classification_report(Y_test, prediction))


params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
#KNeighborsClassifier()
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'], 
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

grid_search=GridSearchCV(cv=5, error_score='raise',
       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy']},
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring=None, verbose=0)
grid_search = GridSearchCV(KNeighborsClassifier(), params, n_jobs = -1, verbose = 1)

#dataframe.drop(dataframe['Volume'], axis = 1).plot()

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
    def __init__(self):
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
        return self.predictors()
    
    def predictors(self):
        dataframe['MA'] = TechnicalIndicators.moving_average(dataframe, 10)
        dataframe['EMA'] = TechnicalIndicators.exponential_moving_average(dataframe, 10)
        dataframe['CCI'] = TechnicalIndicators.commodity_channel_index(dataframe, 10)
        dataframe['MMT'] = TechnicalIndicators.momentum(dataframe, 10)
        dataframe['SOD'] = TechnicalIndicators.stochastic_oscillator_d(dataframe, 10)
        dataframe['SOK'] = TechnicalIndicators.stochastic_oscillator_k(dataframe)
        dataframe['FI'] = TechnicalIndicators.force_index(dataframe, 10)
        dataframe['MI'] = TechnicalIndicators.mass_index(dataframe, 10)
        return dataframe.fillna(0, inplace=True)
        
        
#pred = Predictors()  

class NormalizeData(object):
    def __init__(self, short_price, long_price):
        
        '''
        :short_price: 
                    MA short and Long
        :long_price:
                    MA Long and Short
        '''
        self.short_price = short_price
        self.long_price = long_price
        
        
    def normalizeData(self):
        Xf = pd.DataFrame(MinMaxScaler().fit_transform(dataframe.drop(['Close'], axis = 1).values))
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
            Yf = np.where(dataframe['Close'].rolling(window = self.short_price).mean() > dataframe['Close'].rolling(window = self.long_price).mean(),1,0)
        #Train/Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(Xf, Yf, test_size = 0.3, random_state = False)
        return Xf, Yf, X_train, X_test, Y_train, Y_test
        

Xf, Yf, X_train, X_test, Y_train, Y_test = NormalizeData(7, 15).normalizeData()



class Model():
    
    def __init__(self):
        '''
        Define the Classifiers to be Used for 
        @Classifiers:
                    List of Tuples
        @Pipeline: Channel of Estimators
        @Employ the use of GridSearchCV
        Predicting Returns
        '''
        self.N_NEIGBORS = 10
        self.KERNELS = ['linear', 'rbf']
        self.GAMMA = [0.0001, 0.001, 0.01, 1]
        self.CRITERION = ['gini', 'entropy']
        self.MAX_DEPTH = 5
        self.MAX_FEATURES = ['auto', 'sqrt', 'log2']
        self.N_VALIDATION = 2
        self.N_COMPONENTS = 2
        self.BEST_ACCURACY = 0.0
        self.BEST_CLASSIFIER = 0
        self.BEST_GRIDSEARCH = ''
        
        self.pipe_KNN = Pipeline([('normalizer', StandardScaler()), ('clf', KNeighborsClassifier())])
        self.pipe_KNN_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)),
                              ('clf', KNeighborsClassifier())])
        self.pipe_SVC = Pipeline([('normalizer', StandardScaler()), ('clf', SVC())])
        self.pipe_SVC_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                              ('clf', SVC())])
        self.pipe_GaussianProcessClassifier = Pipeline([('normalizer', StandardScaler()), 
                                                ('clf', GaussianProcessClassifier(1.0 * RBF(1.0)))])
        self.pipe_GaussianProcessClassifier_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                                    ('clf', GaussianProcessClassifier(1.0 * RBF(1.0)))])
        self.pipe_LogisticRegression = Pipeline([('normalizer', StandardScaler()), ('clf', LogisticRegression())])
        self.pipe_LogisticRegression_PCA = Pipeline([('normalizer', StandardScaler()),('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                           ('clf', LogisticRegression())])
        self.pipe_DecisionTreeClassifier = Pipeline([('normalizer', StandardScaler()), ('clf', DecisionTreeClassifier())])
        self.pipe_DecisionTreeClassifier_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                                 ('clf', DecisionTreeClassifier())])
        self.pipe_RandomForestClassifier = Pipeline([('normalizer', StandardScaler()), ('clf', RandomForestClassifier())])
        self.pipe_RandomForestClassifier_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                                 ('clf', RandomForestClassifier())])
        self.pipe_AdaBoostClassifier = Pipeline([('normalizer', StandardScaler()), 
                                         ('clf', AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))])
        self.pipe_AdaBoostClassifier_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                             ('clf', AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))])
        self.pipe_GaussianNB = Pipeline([('normalizer', StandardScaler()), ('clf', GaussianNB())])
        self.pipe_GaussianNB_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                     ('clf', GaussianNB())])
        self.pipe_QuadraticDiscriminantAnalysis = Pipeline([('normalizer', StandardScaler()), 
                                                    ('clf', QuadraticDiscriminantAnalysis())])
        self.pipe_QuadraticDiscriminantAnalysis_PCA = Pipeline([('normalizer', StandardScaler()), 
                                                    ('PCA', PCA(n_components = self.N_COMPONENTS)), ('clf', QuadraticDiscriminantAnalysis())])
        
        self.pipe_KNN_param = [{'clf__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
        
        self.pipe_SVC_params = [{'clf__kernel': self.KERNELS,
                                'clf__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                'clf__gamma': self.GAMMA}]
        
        self.pipe_AdaBoostClassifier_param = [{'clf__n_estimators': np.arange(1,50)}]
        
        
        self.pipe_RandomForestClassifier_params = [{'clf__criterion': self.CRITERION,
                                                     'clf__max_depth': np.arange(2,10),
                                                     'clf__min_samples_split': np.arange(2,10),
                                                     'clf__min_samples_leaf': np.arange(2,10)}]
    
        self.pipe_DecisionTreeClassifier_param = [{'clf__max_depth': np.arange(2,10),
                                                    }]
        
        self.pipe_GaussianNB_params = [{'clf__priors': [None]}]
        
        self.pipe_GaussianProcessClassifier_params = [{'clf__kernel': [1**2 * RBF(1.0)]}]
    
        self.pipe_LogisticRegression_params = [{'clf__penalty': ['l1', 'l2'],
                                        		'clf__C': [1.0, 0.5, 0.1], 'clf__solver': ['liblinear']}]
    
        self.QuadraticDiscriminantAnalysis_params = [{'clf__priors': [None]}]
        
        
        
    def optmimize(self):
        '''
        Here we call the GridSearchCV class to get
        the best parameters or better still optimized parameters
        for our data.
        Remember the Gridsearch is done througk the pipeline.
        '''
        
        self.grid_RandomForestClassifier = GridSearchCV(estimator = self.pipe_RandomForestClassifier, param_grid = self.pipe_RandomForestClassifier_params,
                        			scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_RandomForestClassifier_PCA = GridSearchCV(estimator = self.pipe_RandomForestClassifier_PCA, param_grid = self.pipe_RandomForestClassifier_params,
                			scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_KNN = GridSearchCV(estimator = self.pipe_KNN, param_grid = self.pipe_KNN_param,
                                     scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_KNN_PCA = GridSearchCV(estimator = self.pipe_KNN_PCA, param_grid = self.pipe_KNN_param,
                             scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_SVC = GridSearchCV(estimator = self.pipe_SVC, param_grid = self.pipe_SVC_params,
                                             scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_SVC_PCA = GridSearchCV(estimator = self.pipe_SVC_PCA, param_grid = self.pipe_SVC_params,
                                     scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_GaussianProcessClassifier = GridSearchCV(estimator = self.pipe_GaussianProcessClassifier, 
                                                           param_grid = self.pipe_GaussianProcessClassifier_params,
                                                           scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_GaussianProcessClassifier_PCA = GridSearchCV(estimator = self.pipe_GaussianProcessClassifier_PCA, 
                                                   param_grid = self.pipe_GaussianProcessClassifier_params,
                                                   scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_LogisticRegression = GridSearchCV(estimator = self.pipe_LogisticRegression, 
                                                    param_grid = self.pipe_LogisticRegression_params,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_LogisticRegression_PCA = GridSearchCV(estimator = self.pipe_LogisticRegression_PCA, 
                                                    param_grid = self.pipe_LogisticRegression_params,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_DecisionTreeClassifier = GridSearchCV(estimator = self.pipe_DecisionTreeClassifier,
                                                        param_grid = self.pipe_DecisionTreeClassifier_param,
                                                        scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_DecisionTreeClassifier_PCA = GridSearchCV(estimator = self.pipe_DecisionTreeClassifier_PCA,
                                                        param_grid = self.pipe_DecisionTreeClassifier_param,
                                                        scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_AdaBoostClassifier = GridSearchCV(estimator = self.pipe_AdaBoostClassifier, 
                                                    param_grid = self.pipe_AdaBoostClassifier_param,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_AdaBoostClassifier_PCA = GridSearchCV(estimator = self.pipe_AdaBoostClassifier_PCA, 
                                                    param_grid = self.pipe_AdaBoostClassifier_param,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
        
        
        self.grid_GaussianNB = GridSearchCV(estimator = self.pipe_GaussianNB, param_grid = self.pipe_GaussianNB_params,
                                            scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_GaussianNB_PCA = GridSearchCV(estimator = self.pipe_GaussianNB_PCA, param_grid = self.pipe_GaussianNB_params,
                                            scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_QuadraticDiscriminantAnalysis = GridSearchCV(estimator = self.pipe_QuadraticDiscriminantAnalysis, 
                                                               param_grid = self.QuadraticDiscriminantAnalysis_params,
                                                               scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_QuadraticDiscriminantAnalysis_PCA = GridSearchCV(estimator = self.pipe_QuadraticDiscriminantAnalysis_PCA, 
                                                               param_grid = self.QuadraticDiscriminantAnalysis_params,
                                                               scoring='accuracy',	cv = self.N_VALIDATION)
        
        
        
        self.All_grids = {'grid_RandomForestClassifier': self.grid_RandomForestClassifier,
                          'grid_RandomForestClassifier_PCA': self.grid_RandomForestClassifier_PCA,
                          'grid_KNN': self.grid_KNN, 'grid_KNN_PCA_': self.grid_KNN_PCA,
                          'grid_SVC': self.grid_SVC, 'grid_SVC_PCA': self.grid_SVC_PCA,
                          'grid_GaussianProcessClassifier': self.grid_GaussianProcessClassifier,
                          'grid_GaussianProcessClassifier_PCA': self.grid_GaussianProcessClassifier_PCA,
                          'grid_LogisticRegression': self.grid_LogisticRegression,
                          'grid_LogisticRegression_PCA': self.grid_LogisticRegression_PCA,
                          'grid_DecisionTreeClassifier': self.grid_DecisionTreeClassifier,
                          'grid_DecisionTreeClassifier_PCA': self.grid_DecisionTreeClassifier_PCA,
                          'grid_AdaBoostClassifier': self.grid_AdaBoostClassifier,
                          'grid_AdaBoostClassifier_PCA': self.grid_AdaBoostClassifier_PCA,
                          'grid_GaussianNB': self.grid_GaussianNB, 
                          'grid_GaussianNB_PCA': self.grid_GaussianNB_PCA,
                          'grid_QuadraticDiscriminantAnalysis': self.grid_QuadraticDiscriminantAnalysis,
                          'grid_QuadraticDiscriminantAnalysis_PCA': self.grid_QuadraticDiscriminantAnalysis_PCA}
        
        print('--------------------------------------------------------')
        print('\tPerforming optimization...')
        for classifier_grid_name, classifier_grid in self.All_grids.items():
            print('----------------------------------------------------------')
            print('Estimation {}'.format(classifier_grid_name))
        	# Fit grid search	
            classifier_grid.fit(X_train, Y_train)
        	# Best params
            print('Best params: {}'.format(classifier_grid.best_params_))
        	# Best training data accuracy
            print('Best training accuracy: {}'.format(classifier_grid.best_score_))
        	# Predict on test data with best params
            Y_Prediction = classifier_grid.predict(X_test)
        	# Test data accuracy of model with best params
            print('Test set accuracy score for best params: {}'.format(accuracy_score(Y_test, Y_Prediction)))
        	# Track best (highest test accuracy) model
            print('--------------------------------------------------------------')
            if accuracy_score(Y_test, Y_Prediction) > self.BEST_ACCURACY:
                self.BEST_ACCURACY = accuracy_score(Y_test, Y_Prediction)
                self.BEST_GRIDSEARCH = classifier_grid
                self.BEST_CLASSIFIER = classifier_grid_name
        print('\nClassifier with best test set accuracy: {}'.format(self.BEST_CLASSIFIER))
            


Model().optmimize()       
            
            
            
            
            

        
        
        
#for classifier in self.classifiers.values():
#    self.pipeline.set_params(clf = classifier)
#    scores = cross_validate(self.pipeline, X_train, Y_train, return_train_score=True)
#    print('---------------------------------')
#    print(str(classifier))
#    print('-----------------------------------')
#    for key, values in scores.items():
#            print(key,' mean ', values.mean())
#            print(key,' std ', values.std())
            



#            
#        Grid = GridSearchCV(self.pipe, param_grid = self.param_grid, verbose = 1, cv = 2)
#        print('Optimization in progress...')
#        print('Pipeline', [Names for Names, Classifier in self.estimators.steps])
#        print('Parameters: \n', self.param_grid)
#        if Grid.fit(X_train, Y_train):
#            print('..Training in progress')
#            print('Best Score: {}\n Best Parameter: {}\n Best Estimator: {}'.format(Grid.best_score_, Grid.best_params_, Grid.best_estimator_))
#        prediction = Grid.predict(X_test)
#        print('Accuracy: {}'.format(accuracy_score(Y_test, prediction)))
#        print(confusion_matrix(Y_test, prediction))
#        print(classification_report(Y_test, prediction))
                
Model().optmimization()      
        



from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('clf', KNeighborsClassifier())]
pipe = Pipeline(estimators)
pipe 

#%%

start_date = datetime(2000, 1, 1)
end_date = datetime.now()
dataframe = PredictiveModel("TSLA", "yahoo", start_date, end_date, False).getdata()

c = PredictiveModel("TSLA", "yahoo", start_date, end_date, False).Predictors()

Xf, Yf, X_train, X_test, Y_train, Y_test = normalizeData(7, 12)


#%% grid search and plot of return   
svm = SVC()   
pgrid = dict(kernel = ['linear', 'rbf'], C = [1,10,100,1000],gamma = [1,0.1,0.001,0.0001]) 
grid_search = GridSearchCV(svm, param_grid = pgrid, cv =2)

grid_search.fit(X_train, Y_train)
grid_search.best_estimator_
grid_search.best_params_
prediction = grid_search.predict(X_test)
accuracy_score(Y_test, prediction)
confusion_matrix(Y_test, prediction)
print(classification_report(Y_test, prediction))
split = int(0.7*len(data))

data['Predicted_Signal'] = grid_search.predict(Xf)
data['Stock_returns'] = np.log(data['Close']/data['Close'].shift(1))
Cumulative_Nifty_returns = np.cumsum(data[split:]['Stock_returns'])
data['Startegy_returns'] = data['Stock_returns']* data['Predicted_Signal'].shift(1)
Cumulative_Strategy_returns = np.cumsum(data[split:]['Startegy_returns'])
plt.figure(figsize=(10,5))
plt.plot(Cumulative_Nifty_returns, color='r',label = ' Stock_returns')
plt.plot(Cumulative_Strategy_returns, color='g', label = 'Strategy Returns')
plt.legend()
plt.show()


#%% simple pipeline


pipeline = Pipeline([
    ('normalizer', StandardScaler()), #Step1 - normalize data
    ('clf', GaussianProcessClassifier()) #step2 - classifier
])
classifiers = {'KNN': KNeighborsClassifier(3), 'SVC': SVC(kernel="linear", C=0.025), 'SVC2': SVC(C = 100, gamma = 1, kernel = 'rbf'),
              'GuassianClassifier': GaussianProcessClassifier(1.0 * RBF(1.0)), 'LinearRegression': LinearRegression(), 
              'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=5), 
              'RandomForest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), 'Adaboost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
              'GuassianNB': GaussianNB(), 'QuadraticDiscriminant': QuadraticDiscriminantAnalysis()}

for clfs in classifiers.values():
    pipeline.set_params(clf = clfs)
    scores = cross_validate(pipeline, X_train, Y_train, return_train_score=True)
    print('---------------------------------')
    print(str(clfs))
    print('-----------------------------------')
    for key, values in scores.items():
        print(key,' mean ', values.mean())
        print(key,' std ', values.std())
        
        

'''

Output
---------------------------------
---------------------------------
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='uniform')
-----------------------------------
fit_time  mean  0.0016624927520751953
fit_time  std  0.00047002165459349024
score_time  mean  0.010637601216634115
score_time  std  0.0009403242932146254
test_score  mean  0.7988992623328723
test_score  std  0.011155414960395732
train_score  mean  0.8952810984060985
train_score  std  0.011108474723187396
---------------------------------
SVC(C=0.025, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
-----------------------------------
fit_time  mean  0.01197346051534017
fit_time  std  0.0021621510095822588
score_time  mean  0.003656625747680664
score_time  std  0.0004701902520620762
test_score  mean  0.8044115952051637
test_score  std  0.01363119326402527
train_score  mean  0.8023554804804803
train_score  std  0.004511972619581756
---------------------------------
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=2, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
-----------------------------------
fit_time  mean  0.03623604774475098
fit_time  std  0.0009407176587631548
score_time  mean  0.011303027470906576
score_time  std  0.001244048929762614
test_score  mean  0.748285500230521
test_score  std  0.018894212940344587
train_score  mean  0.9712173712173713
train_score  std  0.004994201488133221
---------------------------------
GaussianProcessClassifier(copy_X_train=True,
             kernel=1**2 * RBF(length_scale=1), max_iter_predict=100,
             multi_class='one_vs_rest', n_jobs=1, n_restarts_optimizer=0,
             optimizer='fmin_l_bfgs_b', random_state=None,
             warm_start=False)
-----------------------------------
fit_time  mean  7.335144440333049
fit_time  std  0.4348486613293609
score_time  mean  0.013962984085083008
score_time  std  0.0
test_score  mean  0.8620014983863532
test_score  std  0.014621933853848832
train_score  mean  0.8834950334950334
train_score  std  0.005942343440752655
---------------------------------
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
-----------------------------------
fit_time  mean  0.003174304962158203
fit_time  std  0.003078574416477895
score_time  mean  0.0006648699442545573
score_time  std  0.00047013408649250444
test_score  mean  0.4399584377144499
test_score  std  0.02273534711025628
train_score  mean  0.45591835178209844
train_score  std  0.010593871476301012
---------------------------------
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
-----------------------------------
fit_time  mean  0.006649096806844075
fit_time  std  0.0009401557209369118
score_time  mean  0.0006647904713948568
score_time  std  0.00047007810231521795
test_score  mean  0.8287085062240664
test_score  std  0.014482322741536015
train_score  mean  0.9036043254793255
train_score  std  0.007438460833006765
---------------------------------
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features=1, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
-----------------------------------
fit_time  mean  0.01296528180440267
fit_time  std  4.052336624139774e-07
score_time  mean  0.0016623338063557942
score_time  std  0.0004700778604684559
test_score  mean  0.8252334024896265
test_score  std  0.016843330456324714
train_score  mean  0.8637250231000232
train_score  std  0.006081724856631408
---------------------------------
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
-----------------------------------
fit_time  mean  0.11437392234802246
fit_time  std  0.004022519306393982
score_time  mean  0.005658864974975586
score_time  std  0.00046937055787496935
test_score  mean  0.8432687874596588
test_score  std  0.010454675926940134
train_score  mean  0.8928555959805959
train_score  std  0.004527378130116225
---------------------------------
GaussianNB(priors=None)
-----------------------------------
fit_time  mean  0.002004702885945638
fit_time  std  0.0008116309673616905
score_time  mean  0.0006621678670247396
score_time  std  0.0004682313206099215
test_score  mean  0.7621109958506224
test_score  std  0.013400773619293291
train_score  mean  0.7649060117810119
train_score  std  0.004449538070466168
---------------------------------
QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
               store_covariance=False, store_covariances=None, tol=0.0001)
-----------------------------------
fit_time  mean  0.002991755803426107
fit_time  std  0.001410851704952674
score_time  mean  0.00033283233642578125
score_time  std  0.00047069600416966455
test_score  mean  0.8127650991240203
test_score  std  0.0072738055772840395
train_score  mean  0.8176101582351581
train_score  std  0.004532473764738439

'''
#%%

'''
--------------------------------------------------------
        Performing optimization...
----------------------------------------------------------
Estimation grid_RandomForestClassifier
Best params: {'clf__criterion': 'gini', 'clf__max_depth': 8, 'clf__min_samples_leaf': 8, 'clf__min_samples_split': 9}
Best training accuracy: 0.855755894590846
Test set accuracy score for best params: 0.8546042003231018
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_RandomForestClassifier_PCA
Best params: {'clf__criterion': 'entropy', 'clf__max_depth': 7, 'clf__min_samples_leaf': 6, 'clf__min_samples_split': 3}
Best training accuracy: 0.7489597780859917
Test set accuracy score for best params: 0.691437802907916
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_KNN
Best params: {'clf__n_neighbors': 10}
Best training accuracy: 0.8037447988904299
Test set accuracy score for best params: 0.778675282714055
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_KNN_PCA_
Best params: {'clf__n_neighbors': 9}
Best training accuracy: 0.7149791955617198
Test set accuracy score for best params: 0.6882067851373183
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_SVC
Best params: {'clf__C': 5, 'clf__gamma': 0.0001, 'clf__kernel': 'linear'}
Best training accuracy: 0.8411927877947295
Test set accuracy score for best params: 0.851373182552504
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_SVC_PCA
Best params: {'clf__C': 1, 'clf__gamma': 1, 'clf__kernel': 'rbf'}
Best training accuracy: 0.7323162274618585
Test set accuracy score for best params: 0.6865912762520194
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_GaussianProcessClassifier
Best params: {'clf__kernel': 1**2 * RBF(length_scale=1)}
Best training accuracy: 0.8585298196948682
Test set accuracy score for best params: 0.8675282714054927
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_GaussianProcessClassifier_PCA
Best params: {'clf__kernel': 1**2 * RBF(length_scale=1)}
Best training accuracy: 0.7295423023578363
Test set accuracy score for best params: 0.7011308562197092
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_LogisticRegression
Best params: {'clf__C': 0.1, 'clf__penalty': 'l1', 'clf__solver': 'liblinear'}
Best training accuracy: 0.8349514563106796
Test set accuracy score for best params: 0.8432956381260097
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_LogisticRegression_PCA
Best params: {'clf__C': 0.1, 'clf__penalty': 'l1', 'clf__solver': 'liblinear'}
Best training accuracy: 0.7267683772538142
Test set accuracy score for best params: 0.7059773828756059
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_DecisionTreeClassifier
Best params: {'clf__max_depth': 3}
Best training accuracy: 0.8280166435506241
Test set accuracy score for best params: 0.8481421647819063
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_DecisionTreeClassifier_PCA
Best params: {'clf__max_depth': 6}
Best training accuracy: 0.7246879334257975
Test set accuracy score for best params: 0.6978998384491115
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_AdaBoostClassifier
Best params: {'clf__n_estimators': 8}
Best training accuracy: 0.8141470180305131
Test set accuracy score for best params: 0.8222940226171244
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_AdaBoostClassifier_PCA
Best params: {'clf__n_estimators': 22}
Best training accuracy: 0.6768377253814147
Test set accuracy score for best params: 0.6348949919224556
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_GaussianNB
Best params: {'clf__priors': None}
Best training accuracy: 0.7441054091539528
Test set accuracy score for best params: 0.7544426494345718
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_GaussianNB_PCA
Best params: {'clf__priors': None}
Best training accuracy: 0.7205270457697642
Test set accuracy score for best params: 0.7075928917609047
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_QuadraticDiscriminantAnalysis
Best params: {'clf__priors': None}
Best training accuracy: 0.7933425797503467
Test set accuracy score for best params: 0.7883683360258481
--------------------------------------------------------------
----------------------------------------------------------
Estimation grid_QuadraticDiscriminantAnalysis_PCA
Best params: {'clf__priors': None}
Best training accuracy: 0.7191400832177531
Test set accuracy score for best params: 0.7075928917609047
--------------------------------------------------------------

Classifier with best test set accuracy: grid_GaussianProcessClassifier
'''