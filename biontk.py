# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 00:11:53 2021

@author: Md Fakrul Islam
"""

import pandas as pd

data_file_path = 'D:/location/dataset_biontek.xlsx'

bio_data = pd.read_excel(data_file_path)

#Material drop Material column
data = bio_data.drop(['Material', 'CellDensity', 'Literature', 'CelllineUsed'], axis = 1)
data.head(5)

print(data.columns)
columns = data.columns

print(type(columns))

columns = columns.tolist()
print(type(columns))

data.dropna(inplace=True)

#Data Plotting Before Scaling
import matplotlib.pyplot as plt
data.plot(kind = 'bar')


from sklearn import preprocessing
x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled, columns = columns)

#Data Plotting After Scaling
data.plot(kind = 'bar')

Y = data['Printability']
Z = data['CellViability']

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import OrderedDict
from datetime import date

%matplotlib inline


data.to_csv('D:/partahabhai/bntk.csv', index=False)


Score=0.0
Coef=0.0 
Intercept=0.0

X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.30, random_state = 150)


#Number of independent Variable
k=len(columns)

#Number of observations
n=len(X_test)

#k- Number of independent Variable
#n- Number of samples
def LRegression(Regressor, k, n):
    
    regression = {}
    
    if (Regressor == 'LinearRegression'):
        
        reg = LinearRegression().fit(X_train, y_train)        
        prediction = reg.predict(X_test)            
        Score = reg.score(X_train, y_train)
        Coef  = reg.coef_
        Intercept = reg.intercept_
        
        #Mean Absolute Error(MAE)
        LMAE = mean_absolute_error(y_test,prediction)
    
        #Mean Squared Error(MSE)
        LMSE = mean_squared_error(y_test,prediction)
    
        #Root Mean Squared Error(RMSE)
        LRMSE = np.sqrt(mean_squared_error(y_test,prediction))
    
        #Root Mean Squared Log Error(RMSLE)
        LLGRMSE = np.log(np.sqrt(mean_squared_error(y_test,prediction)))
    
        #R Squared (R2)
        LRSQRD = r2_score(y_test,prediction)
        n=len(X_test)
        #Number of independent Variable
        k=len(columns)
    
        #Adjusted R Squared
        LARSQRD = 1 - ((1-LRSQRD)*(n-1)/(n-k-1))
        
        
        regression['Regressor']=Regressor
        regression['Mean_Absolute_Error']=LMAE
        regression['Mean_Squared_Error']=LMSE
        regression['Root_Mean_Squared_Error']= LRMSE
        regression['Root_Mean_Squared_Log_Error']=LLGRMSE
        regression['R_Squared']= LRSQRD
        regression['Adjusted_R_Squared']= LARSQRD
        regression['Prediction']= prediction
        
        
        return regression
        
        
def RandomFRegression(Regressor,k,n):
    
    regression = {}
    
    if (Regressor == 'RandomForestRegressor'):
        
        reg = RandomForestRegressor(max_depth=10, n_estimators = 50, random_state=22)
        reg.fit(X_train, y_train)
        prediction = reg.predict(X_test)    
        
        
        #Mean Absolute Error(MAE)
        LMAE = mean_absolute_error(y_test,prediction)
    
        #Mean Squared Error(MSE)
        LMSE = mean_squared_error(y_test,prediction)
    
        #Root Mean Squared Error(RMSE)
        LRMSE = np.sqrt(mean_squared_error(y_test,prediction))
    
        #Root Mean Squared Log Error(RMSLE)
        LLGRMSE = np.log(np.sqrt(mean_squared_error(y_test,prediction)))
    
        #R Squared (R2)
        LRSQRD = r2_score(y_test,prediction)
        n=len(X_test)
        #Number of independent Variable
        k=len(columns)
    
        #Adjusted R Squared
        LARSQRD = 1 - ((1-LRSQRD)*(n-1)/(n-k-1))
        
        regression['Regressor']=Regressor
        regression['Mean_Absolute_Error']=LMAE
        regression['Mean_Squared_Error']=LMSE
        regression['Root_Mean_Squared_Error']= LRMSE
        regression['Root_Mean_Squared_Log_Error']=LLGRMSE
        regression['R_Squared']= LRSQRD
        regression['Adjusted_R_Squared']= LARSQRD
        regression['Prediction']= prediction
        
        
        return regression 

def KNRegressor(Regressor,k,n):
    
    regression = {}
    
    if (Regressor == 'KNeighborsRegressor'):
        
        reg = KNeighborsRegressor(n_neighbors=3)
        reg.fit(X_train, y_train)
        prediction = reg.predict(X_test)    
        
        
        #Mean Absolute Error(MAE)
        LMAE = mean_absolute_error(y_test,prediction)
    
        #Mean Squared Error(MSE)
        LMSE = mean_squared_error(y_test,prediction)
    
        #Root Mean Squared Error(RMSE)
        LRMSE = np.sqrt(mean_squared_error(y_test,prediction))
    
        #Root Mean Squared Log Error(RMSLE)
        LLGRMSE = np.log(np.sqrt(mean_squared_error(y_test,prediction)))
    
        #R Squared (R2)
        LRSQRD = r2_score(y_test,prediction)
        n=len(X_test)
        #Number of independent Variable
        k=len(columns)
    
        #Adjusted R Squared
        LARSQRD = 1 - ((1-LRSQRD)*(n-1)/(n-k-1))
        
        regression['Regressor']=Regressor
        regression['Mean_Absolute_Error']=LMAE
        regression['Mean_Squared_Error']=LMSE
        regression['Root_Mean_Squared_Error']= LRMSE
        regression['Root_Mean_Squared_Log_Error']=LLGRMSE
        regression['R_Squared']= LRSQRD
        regression['Adjusted_R_Squared']= LARSQRD
        regression['Prediction']= prediction
        
        return regression
                

def SRegressor(Regressor, k, n):
    
    regression = {}
    
    if (Regressor == 'SVMRegressor'):

        regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        regr.fit(X_train, y_train)
        prediction = regr.predict(X_test)
        
        
        #Mean Absolute Error(MAE)
        LMAE = mean_absolute_error(y_test,prediction)
    
        #Mean Squared Error(MSE)
        LMSE = mean_squared_error(y_test,prediction)
    
        #Root Mean Squared Error(RMSE)
        LRMSE = np.sqrt(mean_squared_error(y_test,prediction))
    
        #Root Mean Squared Log Error(RMSLE)
        LLGRMSE = np.log(np.sqrt(mean_squared_error(y_test,prediction)))
    
        #R Squared (R2)
        LRSQRD = r2_score(y_test,prediction)
        n=len(X_test)
        #Number of independent Variable
        k=len(columns)
    
        #Adjusted R Squared
        LARSQRD = 1 - ((1-LRSQRD)*(n-1)/(n-k-1))
        
        regression['Regressor']=Regressor
        regression['Mean_Absolute_Error']=LMAE
        regression['Mean_Squared_Error']=LMSE
        regression['Root_Mean_Squared_Error']= LRMSE
        regression['Root_Mean_Squared_Log_Error']=LLGRMSE
        regression['R_Squared']= LRSQRD
        regression['Adjusted_R_Squared']= LARSQRD
        regression['Prediction']= prediction
        
        
        return regression


all_results = []

    
result = LRegression('LinearRegression',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('LinearRegression-Printability')


result = RandomFRegression('RandomForestRegressor',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('RandomForestRegressor-Printability')



result = KNRegressor('KNeighborsRegressor',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('KNeighborsRegressor-Printability')



result = SRegressor('SVMRegressor',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('SVMRegressor-Printability')

regression_dataframe_printability = pd.DataFrame(all_results)
regression_dataframe_printability


Score=0.0
Coef=0.0 
Intercept=0.0

X_train, X_test, y_train, y_test = train_test_split(data, Z, test_size=0.30, random_state = 150)


#Number of independent Variable
k=len(columns)

#Number of observations
n=len(X_test)

all_results=[]    

result = LRegression('LinearRegression',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('LinearRegression-CellViability')


result = RandomFRegression('RandomForestRegressor',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('RandomForestRegressor-CellViability')


result = KNRegressor('KNeighborsRegressor',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('KNeighborsRegressor-CellViability')

result = SRegressor('SVMRegressor',k,n)
all_results.append(result)
plt.scatter(y_test,result['Prediction'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('SVMRegressor-CellViability')

regression_dataframe_cellViability = pd.DataFrame(all_results)
regression_dataframe_cellViability


