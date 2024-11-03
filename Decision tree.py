#!/usr/bin/env python
# coding: utf-8

# In[2]:


# step one import the data

import pandas as pd

df= pd.read_csv(r"C:\Users\17063\Downloads\DataSetForPhishingVSBenignUrl (3).csv")

print(df.head())

print(df.tail())


# In[3]:


import numpy as np
# Step 2a: Encode the Target Column
# Convert 'URL_Type_obf_Type' into binary labels: 1 for phishing, 0 for benign
df['URL_Type_obf_Type'] = df['URL_Type_obf_Type'].apply(lambda x: 0 if x == 'Benign' else 1)

# Step 2b: Handle Missing Values
# Fill missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Step 2c: Handle Infinite Values
# Replace infinite values in 'argPathRatio' with the median of that column
# This avoids issues with large values disrupting the model training
df['argPathRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['argPathRatio'].fillna(df['argPathRatio'].median(), inplace=True)

# Display dataset information to verify changes
print(df.info())


# In[4]:


#Step 3: Splitting the Data

from sklearn.model_selection import train_test_split
 
    # Step 3a: Separate Features and Target
# Drop the target column 'URL_Type_obf_Type' to create features (X) and set target (y)
X = df.drop(columns=['URL_Type_obf_Type'])
y = df['URL_Type_obf_Type']

# Step 3b: Split the Data into Training and Testing Sets
# 70% of the data is used for training, and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shape of the split datasets to verify
print("Training set size:", X_train.shape, y_train.shape)
print("Testing set size:", X_test.shape, y_test.shape)


# In[6]:


#step 4: Model training and evalaution

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


# In[7]:


# Initialize a list to store results for each model configuration
results = []

# Loop through depths from 1 to 6
for depth in range(1, 7):
    # Test both 'gini' and 'entropy' criteria for each depth
    for criterion in ["gini", "entropy"]:
        # Initialize the Decision Tree model with specified depth and criterion
        model = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=42)
        
        # Train the model on the training data
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate accuracy of the model on the test set
        accuracy = accuracy_score(y_test, y_pred)
        
        # Append results to the list
        results.append({
            'Depth': depth,
            'Criterion': criterion,
            'Accuracy': accuracy
        })

# Convert the results to a DataFrame for easy comparison
results_df = pd.DataFrame(results)

# Display the results
print(results_df)


# In[8]:


# step 5:model selction and visualization

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[9]:


# Select the best model for depth 2 based on accuracy
# Assuming that both Gini and Entropy perform similarly for depth 2, we'll use Gini for visualization here
best_model = DecisionTreeClassifier(max_depth=2, criterion="gini", random_state=42)
best_model.fit(X_train, y_train)

# Plot the Decision Tree
plt.figure(figsize=(20, 10))  # Set figure size for readability
plot_tree(best_model, feature_names=X.columns, class_names=["Benign", "Phishing"], filled=True, rounded=True)
plt.title("Decision Tree Visualization for Depth 2 (Gini)")
plt.show()


# In[10]:


#increased the depth to 3

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Initialize the Decision Tree model with a depth of 3 and Gini criterion
deeper_model = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)

# Step 2: Train the model on the training data
deeper_model.fit(X_train, y_train)

# Step 3: Visualize the Decision Tree with depth 3
plt.figure(figsize=(25, 12))  # Adjust figure size for readability
plot_tree(
    deeper_model, 
    feature_names=X.columns, 
    class_names=["Benign", "Phishing"], 
    filled=True, 
    rounded=True
)
plt.title("Decision Tree Visualization for Depth 3 (Gini)")
plt.show()


# In[ ]:




