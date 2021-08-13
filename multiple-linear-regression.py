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
         'State', 'County', 'Zip', 'Lat', 'Lng', 'Interaction', 'TimeZone', 'Additional_charges'], axis=1)

cat_columns = df.select_dtypes(exclude="number").columns

# Give categorical columns a numeric value
for col in cat_columns:
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes

df.head()


# In[3]:


# export prepared data
df.to_csv('data/medical_prepared.csv')


# # Univariate Analysis

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# perform univariate analysis on all columns

for col in df.columns:
    plt.hist(df[col])
    plt.title(col)
    
    path = 'plots/univariate-%s.png'%col
    plt.gcf().savefig(path)
    plt.show()


# # Bivariate Analysis
# 
# Since we are predicting Initial_days we will include Initial_days in our bivariate analysis of the features

# In[6]:


for col in df:
    if col != outcome:
        df.plot(kind='scatter', x=outcome, y=col)
        
        path = 'plots/bivariate-%s-%s.png'%(outcome,col)
        plt.gcf().savefig(path)
        plt.show()


# ## Correlation Matrix

# In[7]:


correl = df.corr()
display(correl)


# In[8]:


abs(df.corr())[outcome].sort_values(ascending=False)


# In[9]:


fig, ax = plt.subplots(figsize=(15,15))
heatmap = sns.heatmap(correl, xticklabels = correl.columns, yticklabels = correl.columns, cmap='RdBu')

heatmap.get_figure().savefig('plots/heatmap.png')


# # Regression Models
# 
# We start with a regression model with all of the features

# In[10]:


import statsmodels.api as sm


# In[11]:


X = df.loc[:,df.columns!=outcome]
y = df[outcome]


# In[ ]:





# In[12]:


Xc = sm.add_constant(X)

initial_model = sm.OLS(y,Xc)
results = initial_model.fit()
results.summary()


# In[ ]:





# ## Data Reduction

# In[13]:


# this section adapted from Chapter4 of "Practical Statistics for Data Scientists" 
# by Bruce, Bruce, and Gedeck

from dmba import backward_elimination, AIC_score
from sklearn.linear_model import LinearRegression

def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(X[variables], y)
    return model

def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(y, model.predict(df[y]))
    return AIC_score(y, model.predict(X[variables]), model)

best_model, best_variables = backward_elimination(X.columns, train_model, score_model, 
                                                verbose=True)

print()
print(f'Intercept: {best_model.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(best_variables, best_model.coef_):
    print(f' {name}: {coef}')


# In[14]:


# New dataset with reduced features
df_reduced = df[best_variables]
df_reduced.head()


# In[15]:


# export reduced data
df_reduced.to_csv('data/medical_reduced.csv')


# In[ ]:





# In[22]:


X_reduced = df_reduced.loc[:,df_reduced.columns!=outcome]
Xc_reduced = sm.add_constant(X_reduced)

model_reduced = sm.OLS(y,Xc_reduced)
results = model_reduced.fit()
results.summary()


# In[17]:





# In[18]:





# ## Residuals

# In[24]:


from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import train_test_split

from yellowbrick.regressor import AlphaSelection, PredictionError, ResidualsPlot


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Ridge()
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
residual = visualizer.poof()

residual.get_figure().savefig('plots/residual-plot.png')


# In[20]:


model = Lasso()
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
prediction_error = visualizer.poof()

prediction_error.get_figure().savefig('plots/prediction_error.png')


# In[ ]:




