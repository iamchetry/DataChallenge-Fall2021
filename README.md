# DataChallenge-Fall2021

This project takes the Dragon dataset from ChemML library and drills down into it by applying the following techniques :

1. LASSO (L1) Regression
2. XGBOOST
3. Random Forest
4. Principal Component Analysis (PCA)
5. K-Means Clustering
6. Plotly (For Interactive Visualization)

The project is developed by using Python 3.9 & IPython notebook.

Libraries utilized :

1. Numpy==1.19.5
2. Pandas==1.3.5
3. Plotly==5.4.0
4. Sklearn==1.0.1
5. XGBOOST==1.5.1
6. Hyperopt==0.2.7
7. Yellowbrick==1.3.post1

Project Structure :
1. data/ : Contains all csv files required for analysis & visualization
2. model_objects/ : Contains pickle objects of ML models
3. scripts/ : Contains .py files for Plotly Functions, Data Creation, Supervised & Unsupervised methods
4. logs_rf.txt : Contains logs while training Random Forest Regressor using Hyper-parameter Tuning
5. logs_xgb.txt : Contains logs while training XGBOOST Regressor using Hyper-parameter Tuning
6. modelling_main.ipynb : Contains Analysis & Visualization
7. data_dump.py : Contains code for importing Dragon dataset from ChemML library and dumping as csv
