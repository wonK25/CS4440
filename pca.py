# -------------------------------------------------------------------------
# AUTHOR: Hyewon Kang
# FILENAME: pca.py
# SPECIFICATION: This program reads a dataset containing several health-related features
#               from 'heart_disease_dataset.csv'. It standardizes all numerical features
#               and performs Principal Component Analysis (PCA) multiple times, each time
#               removing one feature from the dataset. The program then calculates and
#               stores the variance explained by the first principal component (PC1) for
#               each iteration. Finally, it identifies and prints which feature’s removal
#               results in the highest PC1 variance.
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 3 to 5 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df = pd.read_csv('heart_disease_dataset.csv')    # df: shape (n_samples, n_features); all columns are numeric features (no label)

features = df.columns                       # keep original feature names to report which one we removed

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)           # column-wise standardization: each feature -> mean 0, std 1
                                                 # note: we standardize once, then drop one column per iteration from this standardized matrix

#Get the number of features
#--> add your Python code here
num_features = scaled_data.shape[1]              #  num_features = p = number of original features (run p iterations; each uses p-1 features)

# Dictionary to store the results
results = {}                                     # Dictionary mapping each removed feature name to its PC1 explained variance ratio

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = np.delete(scaled_data, i, axis = 1) # axis = 1 (columns) and axis = 0 (rows), remove column i (axis=1); reduced_data has shape (n_samples, p-1)

    # Run PCA on the reduced dataset
    # --> add your Python code here
    pca = PCA(n_components = 1)                  # keep only PC1 (the direction capturing the largest variance)
    pca.fit(reduced_data)                        # PCA learns PC1 from the remaining (p-1) features

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    feature_name = features[i]               # the feature we dropped in this iteration
    pc1_variance = pca.explained_variance_ratio_[0]  # proportion of total variance captured by PC1 (0..1)
    results[feature_name] = pc1_variance          # store the result for later comparison

    #print("Computed PC1 variance after removing each feature:")
    #print(f"{feature_name}: {pc1_variance:.4f}")

#print()
#for feature_name, var in sorted(results.items(), key=lambda x: x[1], reverse=True):
#    print(f"{feature_name}: {var:.4f}")

# Find the maximum PC1 variance
# --> add your Python code here
max_variance_feature = max(results, key = lambda k: results[k]) # argument of maximum over the dictionary by value
max_variance = results[max_variance_feature]                    # the corresponding best PC1 ratio

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {max_variance:.4f} when removing {max_variance_feature}")
print()





