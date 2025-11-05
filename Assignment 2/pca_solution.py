# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here

df = pd.read_csv('heart_disease_dataset.csv', sep=',', header=0)

#Create a training matrix
#--> add your Python code here

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

num_features = df.shape[1]
pc1_variance_values = []

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    # Store PC1 variance
    pc1_variance_values.append((df.columns[i], pca.explained_variance_ratio_[0]))

# Find the maximum PC1 variance using a loop
max_pc1_variance = pc1_variance_values[0]
for item in pc1_variance_values:
    if item[1] > max_pc1_variance[1]:
        max_pc1_variance = item

# Print results
print("PC1 Variance for Each Feature Removed:")
for feature, variance in pc1_variance_values:
    print(f"Removing {feature}: PC1 Variance = {variance:.4f}")

print(f"\nHighest PC1 variance found: {max_pc1_variance[1]:.4f} when removing {max_pc1_variance[0]}.")





