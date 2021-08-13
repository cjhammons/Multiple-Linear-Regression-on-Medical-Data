#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install numpy
# !pip install pandas
# !pip install matplotlib
# !pip install sklearn
# !pip install dmba
# !pip install statsmodels
# !pip install yellowbrick


# In[2]:


import pandas as pd
import numpy as np

df = pd.read_csv("data/medical_clean.csv")
outcome = 'TotalCharge'

df = df.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 
             'State', 'County', 'Zip', 'Lat', 'Lng', 'Interaction', 'TimeZone', 
              'Additional_charges'], axis=1)

cat_columns = df.select_dtypes(exclude="number").columns

# Give categorical columns a numeric value
for col in cat_columns:
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes

df.head()


# In[3]:


# export prepared data
df.to_csv('data/medical_prepared.csv')


# In[4]:


df['Complication_risk']


# # Univariate Analysis

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns



# In[6]:


# perform univariate analysis on all columns

for col in df.columns:
    plt.hist(df[col])
    plt.title(col)
    
    path = 'plots/univariate-%s.png'%col
    plt.gcf().savefig(path)


# # Bivariate Analysis
# 
# Since we are predicting Initial_days we will include Initial_days in our bivariate analysis of the features

# In[7]:


for col in df:
    if col != outcome:
        df.plot(kind='scatter', x=outcome, y=col)
        
        path = 'plots/bivariate-%s-%s.png'%(outcome,col)
        plt.gcf().savefig(path)


# ## Correlation Matrix

# In[8]:


correl = df.corr()
display(correl)


# In[9]:


abs(df.corr())[outcome].sort_values(ascending=False)


# In[10]:


fig, ax = plt.subplots(figsize=(15,15))
heatmap = sns.heatmap(correl, xticklabels = correl.columns, yticklabels = correl.columns, cmap='RdBu')

heatmap.get_figure().savefig('plots/heatmap.png')


# # Regression Models
# 
# We start with a regression model with all of the features

# In[11]:


import statsmodels.api as sm


# In[12]:


X = df.loc[:,df.columns!=outcome]
y = df[outcome]


# In[13]:


Xc = sm.add_constant(X)

initial_model = sm.OLS(y,Xc)
results = initial_model.fit()
results.summary()


# In[ ]:





# ## Data Reduction

# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

linear_regression = LinearRegression(normalize=False,fit_intercept=True)

Xc = sm.add_constant(X)


def r2_est(X,y):
    return r2_score(y,linear_regression.fit(X,y).predict(X))

r2_impact = list()
for j in range(X.shape[1]):
    selection = [i for i in range(X.shape[1]) if i!=j]
    r2_impact.append(((r2_est(Xc,y) - r2_est(Xc.values [:,selection],y)) ,Xc.columns[j]))


best_variables = list()

for imp, varname in sorted(r2_impact, reverse=True):
    if imp >= 0.0005:
        best_variables.append(varname)
    print ('%6.5f %s' %  (imp, varname))

    # New dataset with reduced features
df_reduced = df[best_variables]
df_reduced.head()


# In[ ]:





# In[ ]:





# In[15]:


# export reduced data
df_reduced.to_csv('data/medical_reduced.csv')


# In[ ]:





# In[16]:


X_reduced = df_reduced.loc[:,df_reduced.columns!=outcome]
Xc_reduced = sm.add_constant(X_reduced)

model_reduced = sm.OLS(y,Xc_reduced)
results = model_reduced.fit()
results.summary()


# In[ ]:





# ## Residuals

# In[17]:


from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import train_test_split

from yellowbrick.regressor import AlphaSelection, PredictionError, ResidualsPlot


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Ridge()
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
residual = visualizer.poof()

residual.get_figure().savefig('plots/residual-plot.png')


# In[19]:


model = Lasso()
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
prediction_error = visualizer.poof()

prediction_error.get_figure().savefig('plots/prediction_error.png')


# In[ ]:




