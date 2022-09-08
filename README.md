# STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA [![HitCount](http://hits.dwyl.io/kennedyCzar/https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA.svg)](http://hits.dwyl.io/kennedyCzar/https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA)
Forecasts stock prices using classical machine learning techniques- A time series analysis & modeling. Employ the Use of Predictive Modeling in Machine Learning to Forecast Stock Return. Approach Used by Hedge Funds to Select Tradeable Stocks

![](https://img.shields.io/badge/python-100%25-green.svg)

Objective:

          Predict stock stock price using Technical Indicators as predictors (features).
          Use Supervised Machine Learning Approach to predict stock prices.
          Employ the use of pipeline and GridSearch to select the best model
          Use Final Model to Predict Stock Returns.
          Show plots of stock Return
          Write Deployable script.


Note:
          
          That Every stock has different behaviour and so at every point we may
          have different best performing algorithm. For instance, after much 
          testing Ranform Forest Algorithm perform better for predicting Apple 
          Stocks than any other algo. Guassian process classifier performed 
          better than every other algo at predicting IBM stocks etc.


Indicators/Predictors Used:

        Moving Averages(Also called Rolling mean)
        Commodity Channel Index
        Momentum
        Stochastic Oscillator(D and K)
        Force Index
        Mass Index

        # You can add ass many predictors are desired.
        # Most importantly if you have to do this, you may
        have to consider a feature selection using XGBoost.
                  
How to Use

          >git clone https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA
          Unpak the Files in a project folder
          
          Add File Path to Environment Variable using Spyder PythonPath
          
          Click on Synchronize with Environment.
          
          Restart Spyder.
          
          Report Issue
          

Output

plot of Feature Importance
![Image of FeatureImportance](https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA/blob/master/_IMAGES/Feature_Importance.png)
Gold Stock Retuns
![Image of Regression](https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA/blob/master/_IMAGES/StockGOLD_2018.png)
General Motors stock returns
![Image of Regression](https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA/blob/master/_IMAGES/StockGM_2018.png)
Apple stock returns
![Image of Regression](https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA/blob/master/_IMAGES/StockAAPL_2018.png)
Tesla Stock Returns
![Image of Regression](https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA/blob/master/_IMAGES/StockTSLA_2018.png)
--------------------------------------------------------
Performing optimization...
----------------------------------------------------------

          Estimation grid_RandomForestClassifier
          Best params: {'clf__criterion': 'gini', 'clf__max_depth': 8, 
          'clf__min_samples_leaf': 8, 'clf__min_samples_split': 9}
          Best training accuracy: 0.855755894590846
          Test set accuracy score for best params: 0.8546042003231018
--------------------------------------------------------------

          Estimation grid_RandomForestClassifier_PCA
          Best params: {'clf__criterion': 'entropy', 'clf__max_depth': 7, 
          'clf__min_samples_leaf': 6, 'clf__min_samples_split': 3}
          Best training accuracy: 0.7489597780859917
          Test set accuracy score for best params: 0.691437802907916
--------------------------------------------------------------

          Estimation grid_KNN
          Best params: {'clf__n_neighbors': 10}
          Best training accuracy: 0.8037447988904299
          Test set accuracy score for best params: 0.778675282714055
--------------------------------------------------------------

          Estimation grid_KNN_PCA_
          Best params: {'clf__n_neighbors': 9}
          Best training accuracy: 0.7149791955617198
          Test set accuracy score for best params: 0.6882067851373183
--------------------------------------------------------------

          Estimation grid_SVC
          Best params: {'clf__C': 5, 'clf__gamma': 0.0001, 'clf__kernel': 'linear'}
          Best training accuracy: 0.8411927877947295
          Test set accuracy score for best params: 0.851373182552504
--------------------------------------------------------------

          Estimation grid_SVC_PCA
          Best params: {'clf__C': 1, 'clf__gamma': 1, 'clf__kernel': 'rbf'}
          Best training accuracy: 0.7323162274618585
          Test set accuracy score for best params: 0.6865912762520194
--------------------------------------------------------------

          Estimation grid_GaussianProcessClassifier
          Best params: {'clf__kernel': 1**2 * RBF(length_scale=1)}
          Best training accuracy: 0.8585298196948682
          Test set accuracy score for best params: 0.8675282714054927
--------------------------------------------------------------

          Estimation grid_GaussianProcessClassifier_PCA
          Best params: {'clf__kernel': 1**2 * RBF(length_scale=1)}
          Best training accuracy: 0.7295423023578363
          Test set accuracy score for best params: 0.7011308562197092
--------------------------------------------------------------

          Estimation grid_LogisticRegression
          Best params: {'clf__C': 0.1, 'clf__penalty': 'l1', 'clf__solver': 'liblinear'}
          Best training accuracy: 0.8349514563106796
          Test set accuracy score for best params: 0.8432956381260097
--------------------------------------------------------------

          Estimation grid_LogisticRegression_PCA
          Best params: {'clf__C': 0.1, 'clf__penalty': 'l1', 'clf__solver': 'liblinear'}
          Best training accuracy: 0.7267683772538142
          Test set accuracy score for best params: 0.7059773828756059
--------------------------------------------------------------

          Estimation grid_DecisionTreeClassifier
          Best params: {'clf__max_depth': 3}
          Best training accuracy: 0.8280166435506241
          Test set accuracy score for best params: 0.8481421647819063
--------------------------------------------------------------

          Estimation grid_DecisionTreeClassifier_PCA
          Best params: {'clf__max_depth': 6}
          Best training accuracy: 0.7246879334257975
          Test set accuracy score for best params: 0.6978998384491115
--------------------------------------------------------------

          Estimation grid_AdaBoostClassifier
          Best params: {'clf__n_estimators': 8}
          Best training accuracy: 0.8141470180305131
          Test set accuracy score for best params: 0.8222940226171244
--------------------------------------------------------------

          Estimation grid_AdaBoostClassifier_PCA
          Best params: {'clf__n_estimators': 22}
          Best training accuracy: 0.6768377253814147
          Test set accuracy score for best params: 0.6348949919224556
--------------------------------------------------------------

          Estimation grid_GaussianNB
          Best params: {'clf__priors': None}
          Best training accuracy: 0.7441054091539528
          Test set accuracy score for best params: 0.7544426494345718
--------------------------------------------------------------

          Estimation grid_GaussianNB_PCA
          Best params: {'clf__priors': None}
          Best training accuracy: 0.7205270457697642
          Test set accuracy score for best params: 0.7075928917609047
--------------------------------------------------------------

          Estimation grid_QuadraticDiscriminantAnalysis
          Best params: {'clf__priors': None}
          Best training accuracy: 0.7933425797503467
          Test set accuracy score for best params: 0.7883683360258481
--------------------------------------------------------------

          Estimation grid_QuadraticDiscriminantAnalysis_PCA
          Best params: {'clf__priors': None}
          Best training accuracy: 0.7191400832177531
          Test set accuracy score for best params: 0.7075928917609047
--------------------------------------------------------------

           Classifier with best test set accuracy: grid_GaussianProcessClassifier

# Conclusion

```
You must note that this strategy is trading is a low frequency approach and this 
fits to make steady income over a period of time.
For high Frequency Trading the result of the return is quite high.

GOLD happens to give the most return on applied strategy(As shown in
the graphs above).
Also worthy of mention is the fact that, Random Forest Classifier + PCA 
in most cases performed better for stocks prices with both unsteady and steady rise. 
Followed Next to Adaboost, then Gradientbost Classifier.
In any case, the performance of an algorithm depends on the structure of 
the underlying prices. Its behaviour over a time series.
For different stocks different agorithm perform best.
```

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA/issues)
