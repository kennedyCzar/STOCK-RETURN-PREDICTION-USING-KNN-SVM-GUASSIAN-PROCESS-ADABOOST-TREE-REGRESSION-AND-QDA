# STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA
Forecast stock prices using machine learning approach. A time series analysis. Employ the Use of Predictive Modeling in Machine Learning to Forecast Stock Return. Approach Used by Hedge Funds to Select Tradeable Stocks


Objective:

          Predict stock stock price using Technical Indicators as predictors.
          Use Supervised Machine Learning Approach to predict stock price both on train and test data.
          Employ the use of pipeline to and GridSearch to select the model model
          Use Final Model to Predict Stock Returns.
          SHow plots of stock Return
          Write Deployable script.

How to Use

          >git clone https://github.com/kennedyCzar/STOCK-RETURN-PREDICTION-USING-KNN-SVM-GUASSIAN-PROCESS-ADABOOST-TREE-REGRESSION-AND-QDA
          Unpak the Files in a project folder
          
          Add File Path to Environment Variable using Spyder PythonPath
          
          Click on Synchronize with Environment.
          
          Restart Spyder.
          
          Report Issue
          

Output
          --------------------------------------------------------
        Performing optimization...
          ----------------------------------------------------------
          Estimation grid_RandomForestClassifier
          Best params: {'clf__criterion': 'gini', 'clf__max_depth': 8, 
          'clf__min_samples_leaf': 8, 'clf__min_samples_split': 9}
          Best training accuracy: 0.855755894590846
          Test set accuracy score for best params: 0.8546042003231018
          --------------------------------------------------------------
          ----------------------------------------------------------
          Estimation grid_RandomForestClassifier_PCA
          Best params: {'clf__criterion': 'entropy', 'clf__max_depth': 7, 
          'clf__min_samples_leaf': 6, 'clf__min_samples_split': 3}
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
