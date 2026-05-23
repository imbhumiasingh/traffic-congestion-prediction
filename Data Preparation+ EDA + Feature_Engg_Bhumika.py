#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy
import scipy
import sklearn

print(numpy.__version__)
print(scipy.__version__)
print(sklearn.__version__)


# In[17]:


# -----------------------------------
# 1. EDA + Data Prparation
# -----------------------------------

# -----------------------------------
#  Step 1- IMPORT LIBRARIES
# -----------------------------------

import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import StandardScaler

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv(r'D:\upgrad\Mentornship\Component 2\Trial datasets\Final Dataset_Uber Traffic.csv')

# -----------------------------------
# DISPLAY FIRST 5 ROWS
# -----------------------------------

print(df.head())


# In[18]:


# -----------------------------------
# STEP 2: Check Dataset Information
# -----------------------------------

print(df.info())

# Check missing values
print(df.isnull().sum())


# In[19]:


# -----------------------------------
# STEP 3: Remove Duplicate Rows
# -----------------------------------

df = df.drop_duplicates()

print("Duplicate rows removed successfully")


# In[20]:


print(df.info())


# In[21]:


# -----------------------------------
# STEP 4: Handle Missing Values
# -----------------------------------

# Fill missing event values
df['Event_name'] = df['Event_name'].fillna('No Event')
df['Event_category'] = df['Event_category'].fillna('No Event')

print("Missing values handled successfully")


# In[22]:


# -----------------------------------
# STEP 5: Correct Inconsistencies
# -----------------------------------

# Convert DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Standardize categorical text formatting
if 'Day_Name' in df.columns:
    df['Day_Name'] = df['Day_Name'].str.strip().str.title()


print("Inconsistencies corrected successfully")


# In[23]:


# -----------------------------------
# STEP 6: Standardization
# -----------------------------------

# Select numerical columns
numerical_cols = [
    'Vehicles',
    'Temperature',
    'Precipitation',
    'Wind Speed',
    'Relative Humidity'
]

# Initialize scaler
scaler = StandardScaler()

# Apply standardization
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Standardization completed successfully")


# In[24]:


# -----------------------------------
# STEP 8: Save Cleaned Dataset
# -----------------------------------

df.to_csv("Cleaned_Uber_Traffic.csv", index = False)

print("Cleaned dataset saved successfully")


# In[25]:


# Feature Engineering - Creating new features from raw data
# a) Time-based features

df['DateTime'] = pd.to_datetime(df['DateTime'],dayfirst=True)

df['Hour'] = df['DateTime'].dt.hour
df['Day_of_Week'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month

print(df[['DateTime', 'Hour', 'Day_of_Week', 'Month']].head())


# In[26]:


# -----------------------------------
# Create Lag Features for 4 Junctions
# -----------------------------------

# Sort data properly
# Important because data is arranged junction-wise

df = df.sort_values(by=['Junction','DateTime'])

# -----------------------------------
# Create Lag Features
# -----------------------------------

# Previous hour traffic for same junction

df['Lag_1'] = df.groupby('Junction')['Vehicles'].shift(1)

# Previous day traffic for same junction
# (24 hours earlier)

df['Lag_24'] = df.groupby('Junction')['Vehicles'].shift(24)

# -----------------------------------
# Remove null values generated due to shifting
# -----------------------------------

df.dropna(inplace=True)

# Display engineered features

print(df[['DateTime','Junction','Vehicles','Lag_1','Lag_24']].head())


# In[36]:


# -------------------------------------------
# 2. FEATURE IMPORTANCE & SELECTION
# -------------------------------------------

# -----------------------------
# A) Correlation Analysis
# -----------------------------

# Select only numeric columns
# Excluding ID because it is only an identifier
numeric_df = df.select_dtypes(include=['number']).drop(columns=['ID'])

# Correlation matrix
correlation_matrix = numeric_df.corr()

# Correlation with target variable
print("\nCorrelation with Vehicles:")
print(correlation_matrix['Vehicles'].sort_values(ascending = False))

# ----------------------------------------
# Correlation Heatmap Visualization
# ----------------------------------------

# Import visualization libraries

import matplotlib.pyplot as plt 
import seaborn as sns

# Set figure size
plt.figure(figsize=(12,8))

# Create heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Add title
plt.title("Correlation Heatmap")

# Display plot
plt.show()


# In[39]:


# ------------------------------------------
# B) Random Forest Feature Importance
# ------------------------------------------

# Import library
from sklearn.ensemble import RandomForestRegressor

# Drop unnecessary columns

X = df.drop(columns=['Vehicles','DateTime'])

# Convert categorical columns if present

X = pd.get_dummies(X,drop_first = True)

# Target Variable

y = df['Vehicles']

# Train Random Forest Model

model = RandomForestRegressor(n_estimators = 100, random_state=42)
model.fit(X,y)

# Feature Importance DataFrame

importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': model.feature_importances_
})

# Sort importance values
importance_df = importance_df.sort_values(by = 'Importance', ascending = False)

# Display feature importance
print("\nFeature Importance:")
print(importance_df)


# In[40]:


# -------------------------------------------
# Feature Importance Visualization
# -------------------------------------------

plt.figure(figsize=(12,6))

plt.bar(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xticks(rotation=90)

plt.xlabel('Features')
plt.ylabel('Importance Score')

plt.title('Feature importance using Random Forest')
plt.show()


# In[41]:


#---------------------------
# Final Selected Features
#---------------------------

selected_features = importance_df[importance_df['Importance']>0.01]

print('\nSelected Important Features:')
print(selected_features)


# In[42]:


df.to_csv("Component3_Feature_Engineered_Dataset.csv", index=False)

print("\nFinal dataset saved successfully.")


# In[ ]:




