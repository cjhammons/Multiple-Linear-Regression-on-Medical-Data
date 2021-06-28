#!/usr/bin/env python
# coding: utf-8

# In[ ]:



get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install dmba')
get_ipython().system('pip install statsmodels')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

from dmba import stepwise_selection
from dmba import AIC_score

from dmba import backward_elimination



get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:





# # Data Preparation
# 
# We're working with a lot of categorical and boolean variables, we need to convert them to numeric so that our regression functions can properly injest them

# In[33]:


df = pd.read_csv('data/medical_clean.csv')

predictors = ['Age', 'Gender', 'HighBlood', 'Stroke', 'Complication_risk',
             'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain',
             'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma', 
             'VitD_levels', 'Soft_drink', 'Services', 'Initial_days',
             'vitD_supp', 'Initial_admin']



df['Gender'] = pd.Categorical(df['Gender'])
df['Gender'] = df['Gender'].cat.codes

df['HighBlood'] = pd.Categorical(df['HighBlood'])
df['HighBlood'] = df['HighBlood'].cat.codes

df['Complication_risk'] = pd.Categorical(df['Complication_risk'])
df['Complication_risk'] = df['Complication_risk'].cat.codes

df['Stroke'] = pd.Categorical(df['Stroke'])
df['Stroke'] = df['Stroke'].cat.codes

df['Overweight'] = pd.Categorical(df['Overweight'])
df['Overweight'] = df['Overweight'].cat.codes

df['Arthritis'] = pd.Categorical(df['Arthritis'])
df['Arthritis'] = df['Arthritis'].cat.codes

df['Hyperlipidemia'] = pd.Categorical(df['Hyperlipidemia'])
df['Hyperlipidemia'] = df['Hyperlipidemia'].cat.codes

df['BackPain'] = pd.Categorical(df['BackPain'])
df['BackPain'] = df['BackPain'].cat.codes

df['Anxiety'] = pd.Categorical(df['Anxiety'])
df['Anxiety'] = df['Anxiety'].cat.codes

df['Allergic_rhinitis'] = pd.Categorical(df['Allergic_rhinitis'])
df['Allergic_rhinitis'] = df['Allergic_rhinitis'].cat.codes

df['Reflux_esophagitis'] = pd.Categorical(df['Reflux_esophagitis'])
df['Reflux_esophagitis'] = df['Reflux_esophagitis'].cat.codes

df['Asthma'] = pd.Categorical(df['Asthma'])
df['Asthma'] = df['Asthma'].cat.codes

df['Soft_drink'] = pd.Categorical(df['Soft_drink'])
df['Soft_drink'] = df['Soft_drink'].cat.codes

df['Services'] = pd.Categorical(df['Services'])
df['Services'] = df['Services'].cat.codes

df['Initial_admin'] = pd.Categorical(df['Initial_admin'])
df['Initial_admin'] = df['Initial_admin'].cat.codes

df['Diabetes'] = pd.Categorical(df['Diabetes'])
df['Diabetes'] = df['Diabetes'].cat.codes

outcome = 'ReAdmis'
df[outcome] = pd.Categorical(df[outcome])
df[outcome] = df[outcome].cat.codes


X = df[predictors]
y = df[outcome].values

X.head()


# In[ ]:





# In[34]:


# outcome = 'ReAdmis'
# df[outcome] = pd.Categorical(df[outcome])
# df[outcome] = df[outcome].cat.codes
# df[outcome].head()

# df.ReAdmis = pd.Categorical(df.ReAdmis)
# df.ReAdmis = df.ReAdmis.cat.codes


# # Initial Regression

# In[35]:


model = LinearRegression()
model.fit(X, y)

#model.summary()
# print(f'Intercept: {model.intercept_:.3f}')
# print(f'Coefficients:')
# for name, coef in zip(predictors, model.coef_):
#     print(f' {name}: {coef}')


# ## Assessing the model's accuracy

# In[36]:


from sklearn.metrics import r2_score

fitted = model.predict(X)
r2 = r2_score(y, fitted)
print(f'r2: {r2:.4f}')


# In[37]:




model = linear_model.LinearRegression(normalize=False, fit_intercept=True)

def r2_est(X, y):
    model.fit(X, y)
    fitted = model.predict(X)
    return r2_score(y, fitted)

print ('Baseline R2: %0.3f' %  r2_est(X,y))

r2_impact = list()
for i in range(len(predictors)):
    selection = [j for j in range(len(predictors)) if i != j]
    r2_impact.append(((r2_est(X, y) - r2_est(X.values[:,selection], y)), predictors[i]))
    
for imp, varname in sorted(r2_impact, reverse=True):
    print('%6.3f %s' % (imp, varname))


# In[14]:


# Code based on page Chapter 4: "Regression and Prediction" from "Practical Statistics for Data Scientists"


#X = pd.get_dummies(df[predictors], drop_first=True)
#y = df[outcome]


# In[ ]:





# In[15]:


'''
from dmba import forward_selection

X = pd.get_dummies(df[predictors], drop_first=True)
y = df[outcome]

def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(X[variables], y)
    return model

def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(y, [y.mean()] * len(y), model, df=1)
    return AIC_score(y, model.predict(X[variables]), model)


def score_model(model, variables):
    if len(variables) == 0:
        return r2_score(y, [y.mean()] * len(y))
    fitted = model.predict(X[variables])
    return r2_score(y, fitted)

best_model, best_variables = backward_elimination(X.columns, train_model, score_model, verbose=True)



reduced_predictors = []

print()
print(f'Intercept: {best_model.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(best_variables, best_model.coef_):
    reduced_predictors.append(name)
    print(f' {name}: {coef}')
'''


# Now we rebuild the model with the reduced list of predictors

# In[16]:


'''
model = LinearRegression()
model.fit(df[reduced_predictors], df[outcome])

print(f'Intercept: {model.intercept_:.3f}')
print(f'Coefficients:')
for name, coef in zip(reduced_predictors, model.coef_):
    print(f' {name}: {coef}')
    '''


# In[17]:



'''fitted = model.predict(df[reduced_predictors])
r2 = r2_score(df[outcome], fitted)
print(f'r2: {r2:.4f}')

#aic = AIC_score(y, fitted)'''


# In[ ]:





# In[18]:



#adapted from D208 lesson 3.3
X = df[predictors]
y = df[outcome]
del X['Initial_days']
predictors.remove('Initial_days')

model = linear_model.LinearRegression(normalize=False, fit_intercept=True)

def r2_est(X, y):
    model.fit(X, y)
    fitted = model.predict(X)
    return r2_score(y, fitted)

print ('Baseline R2: %0.3f' %  r2_est(X,y))


# In[19]:




r2_impact = list()
for i in range(len(predictors)):
    selection = [j for j in range(len(predictors)) if i != j]
    r2_impact.append(((r2_est(X, y) - r2_est(X.values[:,selection], y)), predictors[i]))
    
for imp, varname in sorted(r2_impact, reverse=True):
    print('%6.3f %s' % (imp, varname))


# In[ ]:




