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


# In[44]:


# Feature Engineering - Creating new features from raw data
# a) Time-based features

# Convert DateTime column

df['DateTime'] = pd.to_datetime(df['DateTime'],dayfirst=True)

# ----------------------------------------------
# BASIC TIME FEATURES
# ----------------------------------------------

# Hour of day (0–23)

df['Hour'] = df['DateTime'].dt.hour

# Day of week
# Monday = 0, Sunday = 6

df['Day_of_Week'] = (df['DateTime'].dt.dayofweek)

# Month number

df['Month'] = df['DateTime'].dt.month

# Day name

df['Day_Name'] = (df['DateTime'].dt.day_name())

# Week number

df['Week_Number'] = (df['DateTime'].dt.isocalendar().week)

# Quarter of year

df['Quarter'] = (df['DateTime'].dt.quarter)

# ----------------------------------------------
# WEEKEND FEATURE
# ----------------------------------------------

# 1 = Weekend
# 0 = Weekday

# -----------------------------------------------
# TRAFFIC SESSION FEATURE
# -----------------------------------------------
# For congestion pattern analysis

def traffic_session(hour):

    if 5 <= hour < 12:
        return 'Morning'

    elif 12 <= hour < 17:
        return 'Afternoon'

    elif 17 <= hour < 21:
        return 'Evening'

    else:
        return 'Night'

df['Traffic_Session'] = (df['Hour'].apply(traffic_session))

# -----------------------------------------------
# PEAK HOUR FEATURE
# -----------------------------------------------

peak_hours = [7, 8, 9, 17, 18, 19]

df['Peak_Hour_Flag'] = df['Hour'].apply(lambda x: 1 if x in peak_hours else 0)

# -----------------------------------------------
# RUSH HOUR CATEGORY
# -----------------------------------------------

def rush_category(hour):

    if hour in [7, 8, 9]:
        return 'Morning Rush'

    elif hour in [17, 18, 19]:
        return 'Evening Rush'

    else:
        return 'Non Rush'

df['Rush_Hour_Category'] = (df['Hour'].apply(rush_category))

# -----------------------------------------------
# DISPLAY CREATED FEATURES
# -----------------------------------------------

print(df[[
    'DateTime',
    'Junction',
    'Hour',
    'Day_of_Week',
    'Day_Name',
    'Month',
    'Week_Number',
    'Quarter',
    'Traffic_Session',
    'Peak_Hour_Flag',
    'Rush_Hour_Category'
]].head())


# In[45]:


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

# ---------------------------------------------
# Rolling average traffic for previous 3 hours
# ---------------------------------------------

df['Rolling_Mean_3'] = (df.groupby('Junction')['Vehicles'].transform(
        lambda x: x.rolling(window=3).mean()))

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
numeric_df = df.select_dtypes(include=['number']).drop(columns=['ID','Junction'], error='ignore')

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
plt.figure(figsize=(14,10))

# Create heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Add title
plt.title("Correlation Heatmap")

# Display plot
plt.show()


# In[46]:


#--------------------------------------------
# Junction-wise correlation analysis
#--------------------------------------------

for junction in df['Junction'].unique():

    print(f"\nCorrelation Analysis - Junction {junction}")

    junction_df = df[df['Junction'] == junction]

    corr_matrix = junction_df.select_dtypes(
        include=['number']
    ).drop(
        columns=['ID', 'Junction'],
        errors='ignore'
    ).corr()

    print(
        corr_matrix['Vehicles']
        .sort_values(ascending=False)
    )


# In[47]:


# ------------------------------------------
# B) Random Forest Feature Importance
# ------------------------------------------

# Import library
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------
# Convert Junction to categorical
# -----------------------------------

df['Junction'] = df['Junction'].astype(str)

#------------------------------------
# Drop unnecessary columns
#------------------------------------

X = df.drop(columns=['Vehicles','DateTime','ID'], errors='ignore')

#-------------------------------------------
# Convert categorical columns if present
#-------------------------------------------

X = pd.get_dummies(X,drop_first = True)

#---------------------
# Target Variable
#---------------------

y = df['Vehicles']

#------------------------------
# Train Random Forest Model
#------------------------------

model = RandomForestRegressor(n_estimators = 100, random_state=42)
model.fit(X,y)

#----------------------------------
# Feature Importance DataFrame
#----------------------------------

importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': model.feature_importances_
})

#------------------------------
# Sort importance values
#------------------------------

importance_df = importance_df.sort_values(by = 'Importance', ascending = False)

#------------------------------
# Display feature importance
#------------------------------

print("\nFeature Importance:")
print(importance_df)


# In[49]:


# ------------------------------------------
# Feature Importance WITHOUT Rolling Mean
# ------------------------------------------

X2 = df.drop(
    columns=[
        'Vehicles',
        'DateTime',
        'ID',
        'Rolling_Mean_3'
    ],
    errors='ignore'
)

# Convert Junction to categorical

X2['Junction'] = X2['Junction'].astype(str)

# One-hot encoding

X2 = pd.get_dummies(X2, drop_first=True)

# Train model

model2 = RandomForestRegressor(n_estimators=100, random_state=42)

model2.fit(X2, y)

# Feature importance

importance_df2 = pd.DataFrame({'Feature': X2.columns,'Importance': model2.feature_importances_})

importance_df2 = importance_df2.sort_values(by='Importance',ascending=False)

print("\nFeature Importance Without Rolling Mean:")

print(importance_df2.head(15))


# In[50]:


# -------------------------------------------
# Feature Importance Visualization
# -------------------------------------------

# Select Top 15 Important Features

top_features = importance_df.head(15)

# Create figure

plt.figure(figsize=(12,6))

plt.bar(
    top_features['Feature'],
    top_features['Importance']
)

# Rotate labels

plt.xticks(rotation=90)

# Labels & titles

plt.xlabel('Features')
plt.ylabel('Importance Score')

plt.title('Top 15 Feature importance using Random Forest')
plt.show()


# In[54]:


#---------------------------
# Final Selected Features
#---------------------------

# Select features with importance greater than 0.01

selected_features = top_features[top_features['Importance']>=0.001]

# Sort selected features

selected_features = selected_features.sort_values(by='Importance', ascending=False)

# Display selected features

print('\nSelected Important Features:')
print(selected_features)

# -----------------------------------
# Visualization
# -----------------------------------

plt.figure(figsize=(10,7))

plt.barh(
    selected_features['Feature'],
    selected_features['Importance']
)

plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.title('Selected Important Features')

plt.gca().invert_yaxis()

plt.show()


# In[55]:


df.to_csv("Component3_Feature_Engineered_Dataset.csv", index=False)

print("\nFinal dataset saved successfully.")


# In[ ]:




