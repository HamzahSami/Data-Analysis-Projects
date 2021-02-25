
# coding: utf-8

# ## Predicting Insurance Costs
# 
# In[1]:

#import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("/Users/hamzah/Documents/data sets/insurance.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# # Data Cleaning

# In[5]:


df.isnull().sum()


# - No nulls or Nan values. 

# In[6]:


orig_rows = df.shape[0]

#removing outliers
stats = df['charges'].describe()
IQR = stats['75%'] - stats['25%']

#lower fence
lf = stats['25%']-(1.5 * IQR)

#upper fence
uf = stats['75%']+(1.5 * IQR)

df = df[df['charges'] > lf]
df = df[df['charges'] < uf]

new_rows = df.shape[0]
df.shape


# In[7]:


print("We removed "+ str(orig_rows-new_rows) + " outliers from the model.")


# # Exploratory Data Analysis

# In[8]:


df['charges'].plot(kind = 'hist')
plt.ylabel("Insurance Charges")
plt.show()


# - As it turns out, insurance charges are skewed right. This makes sense as insurance charges are normally within a range of values. 

# In[9]:


df['charges'].describe()


# In[10]:


df['region'].unique()


# In[11]:


corrs = df.corr()
corrs


# - There are no significant correlations. If anything, there are weak relationships between the other variables and insurance charges.

# In[12]:


#heatmap
import seaborn as sn
sn.heatmap(corrs, annot=True)
plt.show()


# - There are weak correlations in all these variables. The strongest correlation is between age and insurance charges. This makes sense since a higher age would mean a higher insurance premium. 

# # Machine Learning

# - Using Random Forests to predict Medical Costs
# - Must change all the character variables to numeric constants (booleans)

# In[13]:


df.dtypes


# - Need to change smoker, region, and sex variables to dummy variables.

# In[14]:


# Change gender

# change objects to strings

cleanup_nums = {"sex":     {"male": 1, "female": 0},
                "smoker": {"yes": 1, "no": 0},
                 "region": {"southeast": 1, "northwest": 2, "southwest": 3, "northeast": 4}}

df.replace(cleanup_nums, inplace=True)
df.head()
df.dtypes


# In[15]:


df.head(5)


# ## Random Forests 

# In[16]:


#import relevant packages
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#Feature Labeling
target = df["charges"]


# In[17]:


inputs = df.drop(["charges"], axis = 1)
inputs.head(5)


# In[18]:


inputs_train, inputs_test, target_train, target_test = train_test_split(inputs, target, test_size=0.2, random_state=0)


# In[19]:


#training the algorithm

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(inputs_train,target_train)


# In[20]:


target_pred = rf.predict(inputs_test)


# In[21]:


predictions = pd.DataFrame({'Actual': target_test, 'Predicted': target_pred})


# In[22]:


# Calculate the absolute errors
errors = abs(target_pred - target_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')


# In[23]:


df1 = predictions.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[24]:


print("Accuracy of Model:" , rf.score(inputs_test,target_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(target_test, target_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(target_test, target_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(target_test, target_pred)))


# - The Random Forests model has 61.34% accuracy. The mean squared error is very high which makes sense given how large the target variable is. On that note, we should look into using a model that can account for the right skewedness behavior of the charges variable. Models such as a Box-Cox transformation or a Yeo-Johnson transformation are viable options to use for our data. 

# # Log-Transform Insurance Costs

# In[25]:


df['log_charges'] = np.log(df['charges'])


# In[26]:


df['log_charges'].plot(kind = 'hist')
plt.ylabel("Insurance Charges")
plt.show()


# - From log-transforming the insurance costs variable, we can see that the histogram is more normally distributed.

# In[27]:


new_target = df['log_charges']
inputs_train2, inputs_test2, target_train2, target_test2 = train_test_split(inputs, new_target, test_size=0.2, random_state=0)


# In[28]:


rf.fit(inputs_train2,target_train2)


# In[29]:


target_pred2 = rf.predict(inputs_test2)


# In[30]:


print("Accuracy of Model:" , rf.score(inputs_test2,target_test2))
print('Mean Absolute Error:', metrics.mean_absolute_error(target_test2, target_pred2))  
print('Mean Squared Error:', metrics.mean_squared_error(target_test2, target_pred2))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(target_test2, target_pred2)))


# - From applying a log-transform, we can see that the accuracy of the model rises up to 74.12%. From here, we can apply a Gridsearch to try and bolster the model's accuracy.

# # Tuning Hyperparameters

# In[32]:


from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)


# In[39]:


grid_search.fit(inputs_train2, target_train2)


# In[40]:


grid_search.best_params_


# In[41]:


best_grid = grid_search.best_estimator_


# In[42]:


target_pred3 = best_grid.predict(inputs_test2)


# In[43]:


print("Accuracy of Model:" , best_grid.score(inputs_test2,target_test2))
print('Mean Absolute Error:', metrics.mean_absolute_error(target_test2, target_pred3))  
print('Mean Squared Error:', metrics.mean_squared_error(target_test2, target_pred3))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(target_test2, target_pred3)))


# - From conducting a GridSearch, the accuracy of the model improves to 79.97%. 
