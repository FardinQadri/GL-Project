#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To help with reading and manipulating data
import pandas as pd
import numpy as np

# To help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To split the data
from sklearn.model_selection import train_test_split

# To help with model building
from sklearn.linear_model import LogisticRegression

# To help with feature scaling
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import accuracy_score

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading the dataset

df = pd.read_csv('Employee.csv')


# ### Data Overview

# In[3]:


# First 5 rows of the data set

df.head()


# In[4]:


# Last 5 rows of the data set

df.tail()


# In[5]:


# Checking the number of rows and columns in the train data

df.shape


# There are 4653 rows and 9 columns in the Employee data set.

# In[6]:


# Check the data types of the columns in the dataset

df.info()


# In[7]:


# Check for the missing values in the data

df.isnull().sum()


# There are no missing values in the data set.

# ## Exploratory Data Analysis (EDA)

# In[8]:


# Statistical summary of the numerical columns in the data

df.describe().T


# ## Data Pre-processing

# In[9]:


# Check for duplicate values in the data

df.duplicated().sum()


# There are 1889 duplicate rows.

# In[10]:


# Dropping the duplicates from the data set
df.drop_duplicates(inplace = True)


# In[11]:


# Label Encoder to convert categorical variables into numbers
le = LabelEncoder()
df['Education'] = le.fit_transform(df.Education)
df['Gender'] = le.fit_transform(df.Gender)
df['City'] = le.fit_transform(df.City)
df['EverBenched'] = le.fit_transform(df.EverBenched)


# In[12]:


df


# In[13]:


# Separating Independent and Dependent Varibles

X = df.drop(columns=['LeaveOrNot'])
y = df[['LeaveOrNot']]


# In[14]:


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[15]:


# Data Overview of the X, y dataframes

print("The shape of the X_train is:", X_train.shape)
print("The shape of the X_test is:", X_test.shape)

print("The shape of the y_train is:", y_train.shape)
print("The shape of the y_test is:", y_test.shape)


# In[16]:


# Train a Random Forest Classifier (you can choose any classification model)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[17]:


# Predicting the Test set results
y_pred = model.predict(X_test)

y_pred


# In[18]:


# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Random Forest Classifier is: ', metrics.accuracy_score(y_pred, y_test))

# ### Saving the Model

# In[19]:


# Importing necessary libraries
from joblib import dump, load


# In[20]:


# Saving my model with the name my_model
dump(model, 'my_model.joblib')


# ### Loading the Model

# In[21]:


from joblib import dump, load
my_rf_model = load('my_model.joblib') 

my_rf_model


# ### Making Predictions

# In[22]:


# Making a single prediction
my_rf_model.predict([[1,2017,2,3,24,1,1,2]])


# In[23]:


# Making a multiple predictions
my_rf_model.predict([[1,2017,2,3,24,1,1,2],[1,2016,2,2,52,0,1,2],[0,2012,2,2,32,1,1,1],[1,2015,2,3,42,3,0,1]])


# ### Making a function to give predictions

# In[24]:


from joblib import dump, load

def load_model_predict(model_path, data_points):

    # Loading the model
    my_rf_model = load(model_path) 

    # Creating a DataFrame 
    temp_dict = {"Education":data_points[0],"JoiningYear":data_points[1],"City":data_points[2],"PaymentTier":data_points[3],"Age":data_points[4],"Gender":data_points[5],"EverBenched":data_points[6],"ExperienceInCurrentDomain":data_points[7]}
    df_temp = pd.DataFrame(temp_dict)

    # Predicting on my_data
    predictions = my_rf_model.predict(df_temp)

    return predictions


# In[25]:


# For making 1 prediction
load_model_predict('my_model.joblib', [[1],[2017],[2],[3],[24],[1],[1],[2]])





