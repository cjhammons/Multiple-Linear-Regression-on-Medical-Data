#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install dmba')
get_ipython().system('pip install statsmodels')
get_ipython().system('pip install sklearn')


# In[3]:


import pandas as pd
import numpy as np

df = pd.read_csv("data/medical_clean.csv")
outcome = 'TotalCharge'

df = df.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 
         'State', 'County', 'Zip', 'Lat', 'Lng', 'Interaction', 'TimeZone'], axis=1)

cat_columns = df.select_dtypes(exclude="number").columns

# Give categorical columns a numeric value
for col in cat_columns:
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes

df.head()


# In[6]:


# export prepared data
df.to_csv('data/medical_prepared.csv')


# # Univariate Analysis

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# perform univariate analysis on all columns

for col in df.columns:
    plt.hist(df[col])
    plt.title(col)
    
    path = 'plots/%s.png'%col
    plt.gcf().savefig(path)
    plt.show()


# # Bivariate Analysis
# 
# Since we are predicting Initial_days we will include Initial_days in our bivariate analysis of the features

# In[9]:


for col in df:
    if col != outcome:
        df.plot(kind='scatter', x=outcome, y=col)
        
        path = 'plots/%s-%s.png'%(outcome,col)
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
sns.heatmap(correl, xticklabels = correl.columns, yticklabels = correl.columns, cmap='RdBu')


# # Regression Models
# 
# We start with a regression model with all of the features

# In[20]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dmba import AIC_score


# In[21]:


X = df.loc[:,df.columns!=outcome]
y = df[outcome]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=101)


# In[26]:


model = LinearRegression()
model.fit(X_train, y_train)

print('AIC Score with all variables: %0.3f' % AIC_score(y_test, model.predict(X_test), model))


# ## Data Reduction

# In[27]:


# from sklearn.model_selection import cross_val_score, ShuffleSplit
# from sklearn.metrics import make_scorer
# from sklearn.preprocessing import PolynomialFeatures


# model = LinearRegression()
# model.fit(X_train, y_train)



# cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
# cv_score = cross_val_score(model, X, y, cv=cv)
# print ('Cv score: mean %0.3f std %0.3f' % (np.mean(np.abs(cv_score)), np.std(cv_score)))


# In[29]:


from dmba import backward_elimination

def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(X_train[variables], y_train)
    return model

def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(y_test, [y_test.mean()] * len(y_test), model, df=1)
    return AIC_score(y_test, model.predict(X_test[variables]), model)

best_model, best_variables = backward_elimination(X.columns, train_model, score_model, 
                                                verbose=True)

print()
print(f'Intercept: {best_model.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(best_variables, best_model.coef_):
    print(f' {name}: {coef}')


# In[32]:


df_reduced = df[best_variables]
df_reduced.head()


# In[33]:


df_reduced.to_csv('data/medical_reduced.csv')

