#!/usr/bin/env python
# coding: utf-8

# # Africa Economic, Banking and Systemic Crisis Data
# Data on Economic and Financial crises in 13 African Countries (1860 to 2014)

# Context--
# This dataset is a derivative of Reinhart et. al's Global Financial Stability dataset which can be found online at: https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx
# 
# Content--
# The dataset specifically focuses on the Banking, Debt, Financial, Inflation and Systemic Crises that occurred, from 1860 to 2014, in 13 African countries, including: Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia and Zimbabwe.
# 
# 
# Acknowledgements--
# Reinhart, C., Rogoff, K., Trebesch, C. and Reinhart, V. (2019) Global Crises Data by Country.
# [online] https://www.hbs.edu/behavioral-finance-and-financial-stability/data. Available at: https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx [Accessed: 17 July 2019].
# 
# 
# Inspiration--
# My inspiration stems from two questions: "Which factors are most associated with Systemic Crises in Africa?" And; "At which annual rate of inflation does an Inflation Crisis become a practical certainty?"

# In[1]:


## Importing libraries

import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# importing datasets

crisis = pd.read_csv('african_crises.csv')


# # 1.Understanding Dataset

# In[3]:


crisis.head(10) ## viewing the 1st 10 rows of the data


# In[4]:


crisis.tail(10) ## viewing the last 10 rows


# In[7]:


crisis.info()


# In[8]:


crisis.shape


# In[9]:


## The data have 1059 rows and 14 columns in it


# In[12]:


crisis.columns ## viewing the datasets columns 


# # 2. Data cleaning and manipulation

# In[13]:


crisis.isnull().any()


# In[14]:


## The data have no missing values in it


# In[15]:


crisis.columns


# In[18]:


crisis['banking_crisis'] = crisis['banking_crisis'].replace('crisis', np.nan)
crisis ['banking_crisis'] = crisis['banking_crisis'].fillna(1)
crisis['banking_crisis'] = crisis['banking_crisis'].replace('no_crisis', np.nan)
crisis ['banking_crisis'] = crisis['banking_crisis'].fillna(0)


# In[21]:


## Looking for data correlation

df = crisis.corr()


# In[23]:


df # This shows that correlations


# In[24]:


## Visualising the data correlation

plt.figure(figsize = (20,10))
sns.heatmap(df,cmap = 'coolwarm', annot = True)


# In[25]:


## The maroon colour shows the maximum correlation 
## The blue colour shows the minimum correlation


# In[27]:


crisis.describe() ##This shows data stats


# # "Which factors are most associated with Systemic Crises in Africa?"

# In[29]:


## To get the factors that are associated with systemic crisis we gonna drop all the unwanted coloumns

crisis.head(2)


# In[31]:


crisis.columns


# In[32]:


df2 = crisis.drop(['cc3', 'exch_usd','inflation_annual_cpi','case'],axis = 1)


# Reason why i droped the those columns is becouse a Systemic risk describes an event that can spark a major collapse in a specific industry or the broader economy. 

# In[33]:


df2.head() # the new data without the dropped columns


# In[34]:


df2.shape


# In[35]:


## we can see the new data has 1059 rows and 10 colums 


# In[39]:


df2['country'].unique() ## this shows all the African countries 


# In[41]:


## We gonna see which countries had systemic crisis

Acrisis = df2.loc[df2['systemic_crisis']==1]


# In[43]:


Acrisis['country'].unique()


# In[44]:


## The countries that had systemic crisis are 'Algeria', 'Central African Republic', 'Ivory Coast', 'Egypt',
## 'Kenya', 'Morocco', 'Nigeria', 'Tunisia', 'Zambia' and 'Zimbabwe


# In[47]:


Acrisis['year'].unique()


# In[48]:


## This are all the years that coutries had the systemic crisis 1870, 1990, 1991, 1992, 1976, 
##1977, 1978, 1979, 1980, 1981, 1982,
##1988, 1989, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 1907, 1931,
##1983, 1985, 1986, 1987, 1984, 2009, 2010, 2011, 2012, 2013, 2014,
##2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008


# In[50]:


## The relationship 

sns.pairplot(Acrisis)


# In[52]:


count = Acrisis['country'].value_counts()
country_count = count[:15,]
plt.figure(figsize=(25,10))
sns.barplot(country_count.index, country_count.values, alpha=1)
plt.xticks(rotation= 30,fontsize=13)
plt.yticks(rotation= 0,fontsize=13)
plt.title('Crisis Numbers in Each Countries',fontsize=20)
plt.ylabel('Number of Crisis', fontsize=20)
plt.xlabel('Country', fontsize=20)
plt.show()


# In[53]:


# the above graph shows the number of crisis in each country as we can
#see the central African Republic has more crisis
# Morocco as the less crisis


# In[57]:


Acrisis.sum()


# In[58]:


## The above sum shows that Banking crisis and independence are the most attribution to
## systemic crisis


# In[60]:


## My focus is gonna be on the most associated crisis to systemic crisis

Acrisis = Acrisis[['country', 'year', 'systemic_crisis', 'independence', 'banking_crisis']]


# In[61]:


Acrisis.head()


# In[63]:


df3_corr = Acrisis.corr()
plt.figure(figsize=(10,7))
sns.heatmap(df3_corr, cmap='coolwarm', annot=True)


# In[64]:


# From the above correlation we can see that the systematic is not correlated to itself as it dependents on 
#other crisis in order for it to happen


# # #             Training and testing data

# In[65]:


from sklearn.model_selection import train_test_split


# In[68]:


X = Acrisis[['independence', 'banking_crisis']]
y = Acrisis['systemic_crisis']


# In[71]:


X_train,x_test,Y_train,y_test = train_test_split(X,y, random_state=0)


# In[73]:


print(X_train)
print(x_test)
print(Y_train)
print(y_test)


# Regressions

# In[74]:


from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'
     .format(knnreg.score(X_test, y_test)))


# In[80]:


## Linear Regression

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))


# In[86]:


plt.figure(figsize=(5,4))
plt.hist(X, y)
plt.plot(X, linreg.coef_ * X + linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
plt.show()


# #  "At which annual rate of inflation does an Inflation Crisis become a practical certainty?"

# In[87]:


crisis.head(5)


# In[88]:


## May focus is on the inflation rate

crisis.columns


# In[89]:


inflation = crisis[['inflation_crises','inflation_annual_cpi', 'country', 'year']]


# In[90]:


inflation.head()


# In[92]:


inflation1 = inflation.loc[inflation['inflation_crises']==1]


# In[94]:


inflation1.head()


# In[96]:


inflation1['country'].unique() ## This show the countries that had inflation crisis


# In[97]:


count = inflation1['country'].value_counts()
country_count = count[:15,]
plt.figure(figsize=(25,10))
sns.barplot(country_count.index, country_count.values, alpha=1)
plt.xticks(rotation= 30,fontsize=13)
plt.yticks(rotation= 0,fontsize=13)
plt.title('Inflation Crisis Numbers in Each Countries',fontsize=20)
plt.ylabel('Number of Crisis', fontsize=20)
plt.xlabel('Country', fontsize=20)
plt.show()


# In[98]:


## The graph shows the number of inflations crisis
##Angola had the highest inflation crisis and the South Africa had the lowest


# In[103]:


## Inflation correlation/relationship
plt.figure(figsize = (20,10))
sns.pairplot(inflation1)


# In[104]:


inflation1


# In[116]:


inflation1.describe()


# In[124]:


sns.boxplot(x = 'inflation_crises', y = 'inflation_annual_cpi', data = inflation1)


# In[128]:


fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(inflation['country'],hue=inflation['inflation_crises'],ax=ax)
plt.title("Inflation crises vs no inflation crises")
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)


# # Focusing on my home country South Africa 

# In[164]:


crisis.head(10)


# In[165]:


crisis['country'].unique()


# In[166]:


south_africa = crisis.loc[crisis['country'] == 'South Africa']


# In[167]:


south_africa.head(10)


# In[168]:


## gonna use the usd exchange and the inflation cpi

south_africa.columns


# In[173]:


color_list = ['blue','black']
ax = south_africa.plot('year','exch_usd', kind='bar',figsize=(20,10),width=0.8,color = color_list,edgecolor = None)
plt.title("exchange rate flow from 1999 to 2014", fontsize=16)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


# In[174]:


## The graph shows the exchange rate flows from 1999 to 2014


# In[178]:


color_list = ['red','black']
ax = south_africa.plot('year','inflation_annual_cpi', kind='bar',figsize=(20,8),width=0.8,color = color_list,edgecolor = None)
plt.title("Inflation rate flow from 1999 to 2014", fontsize=16)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


# In[179]:


## The graph shows the inflation flows


# In[183]:


south_africa.tail(20)


# In[184]:


## Analysisng the inflation and exchange from 1994 - 2014

df = south_africa.tail(20)


# In[185]:


color_list = ['blue','black']
ax = df.plot('year','exch_usd', kind='bar',figsize=(20,10),width=0.8,color = color_list,edgecolor = None)
plt.title("exchange rate flow from 1994 to 2014", fontsize=16)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


# In[191]:


color_list = ['red','black']
ax = df.plot('year','inflation_annual_cpi', kind='bar',figsize=(20,10),width=0.8,color = color_list,edgecolor = None)
plt.title("inflation rate flow from 1994 to 2014", fontsize=16)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


# In[209]:


#Relation between the  Country and the exhange rate and inflation rate wrt USA

plt.figure(figsize=[10,5])
sns.boxplot(south_africa["country"],south_africa["exch_usd"]) 
plt.xticks(rotation = 90)
plt.title('boxplot of exchange rate')
plt.show()


# In[210]:


plt.figure(figsize=[10,5])
sns.boxplot(south_africa["country"],south_africa['inflation_annual_cpi']) 
plt.xticks(rotation = 90)
plt.title('boxplot of inflation rate')
plt.show()


# In[245]:


## The above graphs shows the outliers has you can see the exchange rate have outliers that are greater then
##4.5 and the inflation rate have outliers greater than 30 and less than -15

south_africa = south_africa.drop(['country'], axis=1)


# #  Decision tree

# In[246]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from statistics import mean
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report


# In[247]:


### We shall separate the features and the target in the form of X and Y
x = south_africa.drop('banking_crisis',axis = 1)
y = south_africa['banking_crisis']


# In[248]:


#Using Train test split 
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3 , random_state = 0)


# In[249]:


#Checking the shapes of the training and testing data
xtrain.shape


# In[250]:


ytrain.shape


# In[251]:


giniDecisionTree = DecisionTreeClassifier(criterion='gini',random_state = 0,max_depth=3, min_samples_leaf=5)
giniDecisionTree.fit(xtrain,ytrain)


# In[252]:


#Prediction 
giniPred = giniDecisionTree.predict(xtest)


# In[253]:


#Accuracy score 
print('Accuracy Score: ',accuracy_score(ytest, giniPred))


# In[254]:


## we can see the accuracy is 0.9714.....


# In[255]:


#Classification Report 
print('Classification Report')
print(classification_report(ytest, giniPred))


# In[256]:


entropyDecisionTree = DecisionTreeClassifier(criterion='entropy',random_state = 100,max_depth=3, min_samples_leaf=5)


# In[257]:


entropyDecisionTree.fit(xtrain,ytrain)


# In[258]:


#Predictions 
entropyPred = entropyDecisionTree.predict(xtest)


# In[259]:


#Accuracy 
print('Accuracy Score: ',accuracy_score(ytest, entropyPred))


# Model Accuracy with GINI as criterion for Decision Tree = 0.9714...../97.14%
# 
# 
# Model Accuracy with ENTROPY as criterion for Decision Tree = 0.9714.../97.14%

# In[262]:


## The end


# In[ ]:




