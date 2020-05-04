# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:46:52 2020

@author: DEBOSMITA
"""

import pandas as pd
boston = pd.read_csv("G:\\PURDUE UNIVERSITY_IBM_SIMPLILEARN_AIML\\DATA SCIENCE WITH PYTHON _MODULE\\Datasets\\boston.csv")
corr_boston = boston.corr()
import seaborn as sns
sns.heatmap(corr_boston,square=True)

# To check if any missing values in the data
boston.isnull().values.any()
boston.isnull().sum()
boston.isnull().sum()
#Out[2]: 
#crim       0
#zn         0
#indus      0
#chas       0
#nox        0
#rm         0
#age        0
#dis        0
#rad        0
#tax        0
#ptratio    0
#black      0
#lstat      0
#medv       0
#dtype: int64
# implies no missing value in the data
boston.shape
# Out[12]: (506, 14)
boston.info()
boston.describe()
# to check if any missing values in a particular variable 
boston["crim"].isnull().values.any()
boston["crim"].describe()
# Store the summary statistics in a dataset
summary =boston.describe()
# Inputting missing values with mean/median
mean1 = boston.mean()
mean2=boston['crim'].mean()
boston['crim'].fillna(mean2,inplace=True)
### Not required above part as no missing values 
#Step2 : Splitting  x and y variables
boston.columns
x = boston.iloc[:,0:13]
y = boston.loc[:,'medv']
#or alternatively
y = boston['medv']

# Alternatively
# x1 = boston.drop('medv', axis=1).values 
x1 = boston.drop('medv', axis=1)
y1 = boston['medv']

# Splitting training and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Step 3- running model in training data 
from sklearn.linear_model import LinearRegression
lm= LinearRegression()
model_boston= lm.fit(x_train,y_train)
type(model_boston)
model_boston.coef_
model_boston.intercept_
# pair the feature names with the coefficients
list(zip(boston.columns,model_boston.coef_ ))

#Out[56]: 
#[('crim', -0.13079985223475676),
# ('zn', 0.049403023542403925),
# ('indus', 0.0010953504465426385),
# ('chas', 2.705366237074499),
# ('nox', -15.95705044522296),
# ('rm', 3.4139733205163276),
# ('age', 0.0011188767017395649),
# ('dis', -1.4930812354897256),
# ('rad', 0.3644223782882292),
# ('tax', -0.013171815453400695),
# ('ptratio', -0.9523696663886331),
# ('black', 0.011749209151249556),
# ('lstat', -0.5940760892620507)]


# Step4 - Making Predictions
pred_medv= model_boston.predict(x_test)
type(pred_medv)
#Out[49]: numpy.ndarray
type(y_test)
# Out[48]: pandas.core.series.Series
predicted_price = pd.DataFrame(pred_medv, columns =["Predicted_HousePrice"])
# converting the y_test series to a dataframe
observed_price = pd.DataFrame({'Observed_HousePrice': y_test})
observed_price= observed_price.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)
Predicted_Boston =pd.concat([x_test,observed_price,predicted_price],axis=1)

#Validating model - checking accuracy
Predicted_Boston1 = Predicted_Boston
Predicted_Boston1['Deviation']= abs(Predicted_Boston1['Observed_HousePrice']
                          - Predicted_Boston1['Predicted_HousePrice'])/Predicted_Boston1['Observed_HousePrice']
## error
Predicted_Boston1['Deviation'].mean()
## Accuracy
1- Predicted_Boston1['Deviation'].mean()
# Out[54]: 0.8276770103158996

model_boston.conf_int()

# R-square
from sklearn.metrics import r2_score
print(r2_score(observed_price, predicted_price))
print(r2_score(y_test, pred_medv))
# 0.733449214745309
# a unit change in X inceraes the house price to rise by beta coeffs unit


#### Different Model Validation techniques
# 1. Holdout Validation Approach
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
from sklearn.linear_model import LinearRegression
lm= LinearRegression()
model_boston= lm.fit(x_train,y_train)
result = model_boston.score(x_test,y_test)
result2 = lm.score(x_test,y_test)
print('Accuracy :' + str(result*100))
# Accuracy : %2f%73.3449214745309
print('Accuracy : %.2f%%' % (result*100))
# Accuracy : 73.34%

# 2. K -Fold Cross Validation :
x = boston.iloc[:,0:13]
y = boston.loc[:,'medv']
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
kfold = KFold(n_splits=10, random_state=5)
model_kfold = LinearRegression()
results_kfold = model_selection.cross_val_score(model_kfold, x, y, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
# since saw from cross validation , performance is not consistent i.e. the 
# accuracy is not consistent  , hence the model is not stable 


# Calculating the RMSE
import numpy as np

rmsd = np.sqrt(mean_squared_error(observed_price, predicted_price)   
# Out[90]: 1.5971739466799182 


**********************************************************


# •	Calculate the Mean Square Error (MSE)
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(observed_sales, predicted_sales)
# Out[95]: 2.550964615953106

 # OR - NOT SURE WHICH CALCULATION MSE IS CORRECT

MSE = np.mean(pred_sales-y_test)**2
print(MSE)
# 0.00040300864760126125



import numpy as np

rmsd = np.sqrt(mean_squared_error(observed_price, predicted_price)   
# Out[90]: 1.5971739466799182 
  # Lower the rmse(rmsd) is, the better the fit
r2_value = r2_score(observed_price, predicted_price)   
# Out[93]: 0.8984204533332627