# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:01:08 2024

@author: 5558022
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset 
data = pd.read_excel('PredictionsData.xlsx') 

# Display the first few rows of the data
data.head(), data.info()

# Performing Exploratory data analysis (EDA)

# Inspecting the data and generating descriptive statistics

# Generate descriptive statistics for the dataset
descriptive_stats = data.describe()
print(descriptive_stats)

# Output the descriptive statistics to an Excel file
output_file = 'descriptive_statistics.xlsx'
descriptive_stats.to_excel(output_file)

# Check for any missing values in the dataset
missing_values = data.isnull().sum()

# Display the descriptive statistics and missing values
descriptive_stats, missing_values

# Display the full correlation matrix
full_corr_matrix = data.corr()

# Display the correlation matrix
print(full_corr_matrix)

# Initial Visualisations

# Set up the matplotlib figure
plt.figure(figsize=(20, 30))

# Loop through each column in the dataset and create a histogram
for i, column in enumerate(data.columns):
    plt.subplot((len(data.columns) + 3) // 4, 4, i + 1)  # Adjust the grid size for 4 columns per row
    sns.histplot(data[column], kde=False, bins=15, color='purple', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout(pad=3.0)
plt.show()

# Calculate mean and standard deviation
mean = data['Grade'].mean()
std_dev = data['Grade'].std()

# Plotting the histogram with a distribution line (KDE)
plt.figure(figsize=(10, 6))  # Set the figure size
sns.histplot(data['Grade'], bins=15, kde=True, color='green', edgecolor='black', alpha=0.7)

# Plot mean line
plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')

# Plot mean - 1 std deviation line
plt.axvline(mean - std_dev, color='blue', linestyle='--', linewidth=2, label=f'-1 Std Dev: {mean - std_dev:.2f}')

# Plot mean + 1 std deviation line
plt.axvline(mean + std_dev, color='orange', linestyle='--', linewidth=2, label=f'+1 Std Dev: {mean + std_dev:.2f}')

# Add labels and title to the plot
plt.title('Distribution of Grades with KDE, Mean, and Std Deviation')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.legend()

# Display the plot
plt.show()

# Plotting the heatmap for correlation analysis
plt.figure(figsize=(20, 10))
sns.heatmap(full_corr_matrix, annot=True,fmt='.2f', cmap='coolwarm', linewidths=0.5)

# Add title to the heatmap
plt.title('Correlation Heatmap')

# Show the plot
plt.show()

# Plotting pairplot for selected features
selected_features = ['Grade', 'COVID', 'Metric1', 'Metric2', 'Metric3', 'Metric4' , 'Metric5', 'Metric6', 'Metric7']
plt.subplot(4, 1, 3)
sns.pairplot(data[selected_features])
plt.title('Pairplot of Selected Features')

plt.tight_layout()
plt.show()

# Visualize outliers using box plots
plt.figure(figsize=(20, 10))
sns.boxplot(data=data.drop(columns=['Grade']))
plt.xticks(rotation=90)
plt.title('Box Plot of Metrics to Identify Outliers')
plt.show()

# Identify Key Correlations with the target variable
target_variable = 'Grade'
key_correlations = full_corr_matrix[target_variable].sort_values(ascending=False)
print(key_correlations)

# Multicollinearity check for the original dataset

# Define a threshold to identify highly correlated features
threshold = 0.8

# Find pairs of features with a correlation coefficient above the threshold
features_to_drop = set()
for i in range(len(full_corr_matrix.columns)):
    feature1 = full_corr_matrix.columns[i]
    for j in range(i + 1, len(full_corr_matrix.columns)):
        feature2 = full_corr_matrix.columns[j]
        if abs(full_corr_matrix.loc[feature1, feature2]) > threshold:
            # Drop the feature with the lower correlation to the target variable 'Grade'
            if abs(key_correlations[feature1]) > abs(key_correlations[feature2]):
                features_to_drop.add(feature2)
            else:
                features_to_drop.add(feature1)
                
# Keep features that are not in the drop list
features_to_keep = [feature for feature in full_corr_matrix.columns if feature not in features_to_drop]

# Display features to drop and those to keep
print("Features to drop due to multicollinearity:")
print(features_to_drop)

# Filter the dataset to only keep the selected features
filtered_data1 = data[features_to_keep]

# Display the features kept
print("Features to keep after removing multicollinearity:")
print(features_to_keep)

# Recalculate the correlation matrix for the filtered dataset
filtered_corr_matrix = filtered_data1.corr()

# Display the correlation matrix
print("Correlation matrix of the filtered dataset:")
print(filtered_corr_matrix)

# Check if any correlations are still above the threshold
high_corr_pairs = []
for i in range(len(filtered_corr_matrix.columns)):
    feature1 = filtered_corr_matrix.columns[i]
    for j in range(i + 1, len(filtered_corr_matrix.columns)):
        feature2 = filtered_corr_matrix.columns[j]
        if abs(filtered_corr_matrix.loc[feature1, feature2]) > threshold:
            high_corr_pairs.append((feature1, feature2, filtered_corr_matrix.loc[feature1, feature2]))
                
print("High correlation pairs remaining after filtering:")
print(high_corr_pairs)

# Output the new set of features
print("Final features after removing multicollinearity:")
print(filtered_data1.columns)

# The removal of the metrics is loss of information hence the PCA approach is used later so that information is not lost and removal of the multi-collinearity

# Standardize the features filtered_data1
scaler = StandardScaler()
X = filtered_data1.drop('Grade', axis=1)
y = filtered_data1['Grade']
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression model on the reduced feature set
lr_model_reduced = LinearRegression()
lr_model_reduced.fit(X_train1, y_train1)

# Make predictions
y_pred_reduced = lr_model_reduced.predict(X_test1)

# Evaluate the model
mse_reduced = mean_squared_error(y_test1, y_pred_reduced)
r2_reduced = r2_score(y_test1, y_pred_reduced)
mae_reduced = mean_absolute_error(y_test1, y_pred_reduced)

print(f'Mean Squared Error (Reduced Features): {mse_reduced}')
print(f'R^2 Score (Reduced Features): {r2_reduced}')
print (f'Mean Absolute Error (Reduced Features): {mae_reduced}')


# Approach 1 - Original dataset

# Define the columns for predicted Maths, English grades, and the target Engineering grade
predicted_math_col = 'Metric27'
predicted_english_col = 'Metric12'   
engineering_grade_col = 'Grade'  

# Standardize the features full dataset
scaler = StandardScaler()
X = data.drop(columns=[engineering_grade_col])
y = data[engineering_grade_col]
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = split_data(X_scaled, y)

# Train and evaluate model function
def train_evaluate_model_with_mae(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

# Initialize a list to store results
results_original = []


# Linear Regression on original data
mse_lr, r2_lr, mae_lr = train_evaluate_model_with_mae(LinearRegression(), X_train, y_train, X_test, y_test)
results_original.append(('Linear Regression', 'original Data', mse_lr, r2_lr, mae_lr))

# Lasso Regression on original data
mse_lasso, r2_lasso, mae_lasso = train_evaluate_model_with_mae(Lasso(), X_train, y_train, X_test, y_test)
results_original.append(('Lasso Regression', 'original Data', mse_lasso, r2_lasso, mae_lasso))

# Ridge Regression on original data
mse_ridge, r2_ridge, mae_ridge = train_evaluate_model_with_mae(Ridge(), X_train, y_train, X_test, y_test)
results_original.append(('Ridge Regression', 'original Data', mse_ridge, r2_ridge, mae_ridge))

# Random Forest on original data
mse_rf, r2_rf, mae_rf = train_evaluate_model_with_mae(RandomForestRegressor(), X_train, y_train, X_test, y_test)
results_original.append(('Random Forest', 'original Data', mse_rf, r2_rf, mae_rf))

# Implementing PCA fo original data
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = split_data(X_pca, y)

# Linear Regression on PCA-transformed original data
mse_lr_pca, r2_lr_pca, mae_lr_pca = train_evaluate_model_with_mae(LinearRegression(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_original.append(('Linear Regression', 'original Data PCA', mse_lr_pca, r2_lr_pca, mae_lr_pca))

# Lasso Regression on PCA-transformed original data
mse_lasso_pca, r2_lasso_pca, mae_lasso_pca = train_evaluate_model_with_mae(Lasso(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_original.append(('Lasso Regression', 'original Data PCA', mse_lasso_pca, r2_lasso_pca, mae_lasso_pca))

# Ridge Regression on PCA-transformed original data
mse_ridge_pca, r2_ridge_pca, mae_ridge_pca = train_evaluate_model_with_mae(Ridge(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_original.append(('Ridge Regression', 'original Data PCA', mse_ridge_pca, r2_ridge_pca, mae_ridge_pca))

# Random Forest on PCA-transformed original data
mse_rf_pca, r2_rf_pca, mae_rf_pca = train_evaluate_model_with_mae(RandomForestRegressor(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_original.append(('Random Forest', 'original Data PCA', mse_rf_pca, r2_rf_pca, mae_rf_pca))

# Effect of Feature Engineering on original data

# Define the standardize_features function
def standardize_features(data, target_column):
    scaler = StandardScaler()
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


# Split data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train, evaluate model, and calculate MSE, R^2, and MAE
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

# Feature engineering
print("Evaluating models on feature-engineered raw data")

# Polynomial Feature engineering
numeric_features = data.select_dtypes(include=['int64', 'float64']).drop(columns=['Grade', 'COVID']).columns
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(data[numeric_features])
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(numeric_features))

# Calculate the skewness for all numeric features in the dataset
skewness = data.skew().sort_values(ascending=False)

# Display the skewness values
print(skewness)

# Set a threshold for skewness to determine which features to log transform
threshold = 0.5

# Identify the features that have skewness greater than the threshold
skewed_features = skewness[(skewness > threshold) | (skewness < -threshold)].index

# Display the identified skewed features
print(f"Features with skewness greater than {threshold} or less than {-threshold}:")
print(skewed_features)

# Visualize the distribution of the skewed features
plt.figure(figsize=(14, len(skewed_features) * 4))

for i, feature in enumerate(skewed_features):
    plt.subplot(len(skewed_features), 1, i + 1)
    sns.histplot(data[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature} (Skewness: {skewness[feature]:.2f})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Apply log transformation to skewed features i have not used log(x) since there are 0s in the COVID column and to avoid error log(1+x) is used.
data_log_transformed = data.copy()
data_log_transformed[skewed_features] = data_log_transformed[skewed_features].apply(lambda x: np.log1p(x))

# Set up the figure and axes for subplots the distribution before and after the log transformation
num_features = len(skewed_features)
fig, axes = plt.subplots(num_features, 2, figsize=(14, 4 * num_features))

# Loop through each skewed feature and create the plots
for i, feature in enumerate(skewed_features):
    # Original feature distribution
    sns.histplot(data[feature], bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Original {feature} Distribution')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')
    
    # Log-transformed feature distribution
    sns.histplot(data_log_transformed[feature], bins=30, kde=True, ax=axes[i, 1])
    axes[i, 1].set_title(f'Log-Transformed {feature} Distribution')
    axes[i, 1].set_xlabel('Value')
    axes[i, 1].set_ylabel('Frequency')

# Adjust the layout for better readability
plt.tight_layout()
plt.show()

# Combine all engineered features with the original dataset
X_combined = pd.concat([X_poly_df, data_log_transformed[skewed_features]], axis=1)
X_combined['Grade'] = data['Grade']
X_combined['COVID'] = data['COVID']

X_scaled_combined, y_combined = standardize_features(X_combined, 'Grade')
X_train_comb, X_test_comb, y_train_comb, y_test_comb = split_data(X_scaled_combined, y_combined)

# Linear Regression on combined features
mse_comb, r2_comb, mae_comb = train_evaluate_model(LinearRegression(), X_train_comb, y_train_comb, X_test_comb, y_test_comb) # Added mae_comb to store the Mean Absolute Error returned by the function
print(f'Linear Regression (Combined Features) - MSE: {mse_comb}, R^2: {r2_comb}, MAE: {mae_comb}') # Print the MAE value
results_original.append(('Linear Regression', 'Combined Features', mse_comb, r2_comb, mae_comb)) # Append MAE to the results

# Lasso Regression on combined features
lasso_regressor = Lasso()
mse_lasso_comb, r2_lasso_comb, mae_lasso_comb = train_evaluate_model(lasso_regressor, X_train_comb, y_train_comb, X_test_comb, y_test_comb) # Added mae_lasso_comb
print(f'Lasso Regression (Combined Features) - MSE: {mse_lasso_comb}, R^2: {r2_lasso_comb}, MAE: {mae_lasso_comb}') # Print the MAE value
results_original.append(('Lasso Regression', 'Combined Features', mse_lasso_comb, r2_lasso_comb, mae_lasso_comb)) # Append MAE to the results

# Ridge Regression on combined features
ridge_regressor = Ridge()
mse_ridge_comb, r2_ridge_comb, mae_ridge_comb = train_evaluate_model(ridge_regressor, X_train_comb, y_train_comb, X_test_comb, y_test_comb) # Added mae_ridge_comb
print(f'Ridge Regression (Combined Features) - MSE: {mse_ridge_comb}, R^2: {r2_ridge_comb}, MAE: {mae_ridge_comb}') # Print the MAE value
results_original.append(('Ridge Regression', 'Combined Features', mse_ridge_comb, r2_ridge_comb, mae_ridge_comb)) # Append MAE to the results

# Random Forest on combined features
rf_model_comb = RandomForestRegressor()
mse_rf_comb, r2_rf_comb, mae_rf_comb = train_evaluate_model(rf_model_comb, X_train_comb, y_train_comb, X_test_comb, y_test_comb) # Added mae_rf_comb
print(f'Random Forest (Combined Features) - MSE: {mse_rf_comb}, R^2: {r2_rf_comb}, MAE: {mae_rf_comb}') # Print the MAE value
results_original.append(('Random Forest', 'Combined Features', mse_rf_comb, r2_rf_comb, mae_rf_comb)) # Append MAE to the results

# Convert the results into a DataFrame
results_original_df = pd.DataFrame(results_original, columns=[
    'Model', 'Data Type',
    'Test MSE', 'Test R^2', 'Test MAE'
])

print(results_original_df)

# Output the results to an Excel file
output_file = 'Model_results_Original_data.xlsx'
results_original_df.to_excel(output_file, index=False)


# Approach 2 - Normalisation of the original dataset for data analysis

# Define the standardize_features function
def standardize_features(data, target_column):
    scaler = StandardScaler()
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Split data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train, evaluate model, and calculate MSE, R^2, and MAE for both training and test data
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    # Predictions on the training set
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    
    # Predictions on the testing set
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    return mse_train, r2_train, mae_train, mse_test, r2_test, mae_test

# Copy the original dataset to avoid modifying it directly
data_normalized = data.copy()

# Identify numeric columns (excluding the 'COVID' column)
numeric_columns = data_normalized.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove('COVID')  # Exclude the 'COVID' column itself from normalization

# Iterate over each numeric column and normalize it
for col in numeric_columns:
    # Calculate average for COVID and non-COVID groups
    average_covid = data_normalized.loc[data_normalized['COVID'] == 1, col].mean()
    average_non_covid = data_normalized.loc[data_normalized['COVID'] == 0, col].mean()
    
    # Calculate the COVID bonus for this column
    covid_bonus = average_covid - average_non_covid
    
    print(f'Average {col} (COVID): {average_covid}')
    print(f'Average {col} (Non-COVID): {average_non_covid}')
    print(f'COVID Bonus for {col}: {covid_bonus}')

    # Normalize the COVID-affected students' data in this column
    data_normalized.loc[data_normalized['COVID'] == 1, col] -= covid_bonus

    # Recalculate and print new averages after normalization
    new_average_covid = data_normalized[data_normalized['COVID'] == 1][col].mean()
    new_average_non_covid = data_normalized[data_normalized['COVID'] == 0][col].mean()

    print(f'New Average {col} (COVID): {new_average_covid}')
    print(f'New Average {col} (Non-COVID): {new_average_non_covid}')

# Print the first few rows of the normalized dataset to verify
print(data_normalized.head())

# Save the normalized data to an Excel file
output_file = 'normalized_data.xlsx'
data_normalized.to_excel(output_file, index=False)

# Define the columns for predicted Maths, English grades, and the target Engineering grade
predicted_math_col = 'Metric27'
predicted_english_col = 'Metric12'
engineering_grade_col = 'Grade'

# Benchmark Calculation - Simple Average of Maths and English
data_normalized['Simple_Average'] = data_normalized[[predicted_math_col, predicted_english_col]].mean(axis=1)

# Calculate MSE, R^2, and MAE for the benchmark
simple_avg_mse = mean_squared_error(data_normalized[engineering_grade_col], data_normalized['Simple_Average'])
simple_avg_r2 = r2_score(data_normalized[engineering_grade_col], data_normalized['Simple_Average'])
simple_avg_mae = mean_absolute_error(data_normalized[engineering_grade_col], data_normalized['Simple_Average'])

# Print the results
print(f'Simple Average Benchmark - MSE: {simple_avg_mse}')
print(f'Simple Average Benchmark - R^2: {simple_avg_r2}')
print(f'Simple Average Benchmark - MAE: {simple_avg_mae}')

# Drop 'Simple_Average' from the data before further processing
data_normalized = data_normalized.drop(columns=['Simple_Average'])

# Standardizing and splitting the normalized data
X_scaled_normalized, y_normalized = standardize_features(data_normalized, 'Grade')
X_train_norm, X_test_norm, y_train_norm, y_test_norm = split_data(X_scaled_normalized, y_normalized)

# Combine the standardized features with the target variable
X_scaled_normalized_df = pd.DataFrame(X_scaled_normalized, columns=data_normalized.drop(columns=['Grade']).columns)
combined_df = pd.concat([X_scaled_normalized_df, y_normalized.reset_index(drop=True)], axis=1)

# Calculate the correlation matrix, including the target variable 'Grade'
corr_matrix_combined = combined_df.corr()

# Display the correlation matrix
print("Correlation Matrix including the Target Variable 'Grade':")
print(corr_matrix_combined)

# Define a threshold for high correlation
threshold = 0.8

# Find pairs of features with a correlation coefficient above the threshold
high_corr_pairs = []
for i in range(len(corr_matrix_combined.columns)):
    for j in range(i + 1, len(corr_matrix_combined.columns)):
        if abs(corr_matrix_combined.iloc[i, j]) > threshold:
            feature1 = corr_matrix_combined.columns[i]
            feature2 = corr_matrix_combined.columns[j]
            high_corr_pairs.append((feature1, feature2, corr_matrix_combined.iloc[i, j]))

# Display the pairs of highly correlated features, including correlations with 'Grade'
print("Highly Correlated Feature Pairs (Correlation > 0.8) Including the Target Variable:")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: Correlation = {pair[2]:.2f}")

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature including the target variable
vif_data_combined = pd.DataFrame()
vif_data_combined["Feature"] = combined_df.columns
vif_data_combined["VIF"] = [variance_inflation_factor(combined_df.values, i) for i in range(combined_df.shape[1])]

# Display the VIF results
print("Variance Inflation Factor (VIF) including the Target Variable 'Grade':")
print(vif_data_combined)

# Initialised the results
results_2 = []

# Implementing PCA for the normalised dataset

# Perform PCA with the goal of retaining 95% of the variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled_normalized)

# Number of components retained
n_components_retained = pca.n_components_
print(f'Number of principal components retained: {n_components_retained}')

# Calculate explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plotting Scree plot to visualise the component selection
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()

# X_scaled_normalized was created from a DataFrame called 'data_normalized'
original_feature_names = data_normalized.drop(columns=['Grade']).columns

# Create a DataFrame to display the loadings of the features in the principal components
loadings_df = pd.DataFrame(pca.components_, columns=original_feature_names)
loadings_df.index = [f'PC{i+1}' for i in range(pca.n_components_)]

# Display the loadings
print("Loadings of the different features for each principal component:")
print(loadings_df)

# Output the loadings_df to an Excel file
output_path = 'pca_loadings_actual.xlsx'
loadings_df.to_excel(output_path)

# Running models on the normalised data

# Linear Regression on normalized data
mse_lr_train_norm, r2_lr_train_norm, mae_lr_train_norm, mse_lr_test_norm, r2_lr_test_norm, mae_lr_test_norm = train_evaluate_model(LinearRegression(), X_train_norm, y_train_norm, X_test_norm, y_test_norm)
results_2.append(('Linear Regression', 'Normalized', mse_lr_train_norm, r2_lr_train_norm, mae_lr_train_norm, mse_lr_test_norm, r2_lr_test_norm, mae_lr_test_norm))
print(f'Linear Regression (Normalized) - Train MSE: {mse_lr_train_norm}, Test MSE: {mse_lr_test_norm}, Train R^2: {r2_lr_train_norm}, Test R^2: {r2_lr_test_norm}, Train MAE: {mae_lr_train_norm}, Test MAE: {mae_lr_test_norm}')

# Lasso Regression on normalized data
mse_lasso_train_norm, r2_lasso_train_norm, mae_lasso_train_norm, mse_lasso_test_norm, r2_lasso_test_norm, mae_lasso_test_norm = train_evaluate_model(Lasso(), X_train_norm, y_train_norm, X_test_norm, y_test_norm)
results_2.append(('Lasso Regression', 'Normalized', mse_lasso_train_norm, r2_lasso_train_norm, mae_lasso_train_norm, mse_lasso_test_norm, r2_lasso_test_norm, mae_lasso_test_norm))
print(f'Lasso Regression (Normalized) - Train MSE: {mse_lasso_train_norm}, Test MSE: {mse_lasso_test_norm}, Train R^2: {r2_lasso_train_norm}, Test R^2: {r2_lasso_test_norm}, Train MAE: {mae_lasso_train_norm}, Test MAE: {mae_lasso_test_norm}')

# Ridge Regression on normalized data
mse_ridge_train_norm, r2_ridge_train_norm, mae_ridge_train_norm, mse_ridge_test_norm, r2_ridge_test_norm, mae_ridge_test_norm = train_evaluate_model(Ridge(), X_train_norm, y_train_norm, X_test_norm, y_test_norm)
results_2.append(('Ridge Regression', 'Normalized', mse_ridge_train_norm, r2_ridge_train_norm, mae_ridge_train_norm, mse_ridge_test_norm, r2_ridge_test_norm, mae_ridge_test_norm))
print(f'Ridge Regression (Normalized) - Train MSE: {mse_ridge_train_norm}, Test MSE: {mse_ridge_test_norm}, Train R^2: {r2_ridge_train_norm}, Test R^2: {r2_ridge_test_norm}, Train MAE: {mae_ridge_train_norm}, Test MAE: {mae_ridge_test_norm}')

# Random Forest on normalized data
mse_rf_train_norm, r2_rf_train_norm, mae_rf_train_norm, mse_rf_test_norm, r2_rf_test_norm, mae_rf_test_norm = train_evaluate_model(RandomForestRegressor(), X_train_norm, y_train_norm, X_test_norm, y_test_norm)
results_2.append(('Random Forest', 'Normalized', mse_rf_train_norm, r2_rf_train_norm, mae_rf_train_norm, mse_rf_test_norm, r2_rf_test_norm, mae_rf_test_norm))
print(f'Random Forest (Normalized) - Train MSE: {mse_rf_train_norm}, Test MSE: {mse_rf_test_norm}, Train R^2: {r2_rf_train_norm}, Test R^2: {r2_rf_test_norm}, Train MAE: {mae_rf_train_norm}, Test MAE: {mae_rf_test_norm}')

# Split the PCA-transformed data into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_normalized, test_size=0.2, random_state=42)

# Linear Regression on PCA-transformed normalised data
mse_lr_train_pca, r2_lr_train_pca, mae_lr_train_pca, mse_lr_test_pca, r2_lr_test_pca, mae_lr_test_pca = train_evaluate_model(LinearRegression(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_2.append(('Linear Regression', 'PCA_Normalised', mse_lr_train_pca, r2_lr_train_pca, mae_lr_train_pca, mse_lr_test_pca, r2_lr_test_pca, mae_lr_test_pca))
print(f'Linear Regression (PCA_Normalised) - Train MSE: {mse_lr_train_pca}, Test MSE: {mse_lr_test_pca}, Train R^2: {r2_lr_train_pca}, Test R^2: {r2_lr_test_pca}, Train MAE: {mae_lr_train_pca}, Test MAE: {mae_lr_test_pca}')

# Lasso Regression on PCA-transformed normalised data
mse_lasso_train_pca, r2_lasso_train_pca, mae_lasso_train_pca, mse_lasso_test_pca, r2_lasso_test_pca, mae_lasso_test_pca = train_evaluate_model(Lasso(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_2.append(('Lasso Regression', 'PCA_Normalised', mse_lasso_train_pca, r2_lasso_train_pca, mae_lasso_train_pca, mse_lasso_test_pca, r2_lasso_test_pca, mae_lasso_test_pca))
print(f'Lasso Regression (PCA_Normalised) - Train MSE: {mse_lasso_train_pca}, Test MSE: {mse_lasso_test_pca}, Train R^2: {r2_lasso_train_pca}, Test R^2: {r2_lasso_test_pca}, Train MAE: {mae_lasso_train_pca}, Test MAE: {mae_lasso_test_pca}')

# Ridge Regression on PCA-transformed normalised data
mse_ridge_train_pca, r2_ridge_train_pca, mae_ridge_train_pca, mse_ridge_test_pca, r2_ridge_test_pca, mae_ridge_test_pca = train_evaluate_model(Ridge(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_2.append(('Ridge Regression', 'PCA_Normalised', mse_ridge_train_pca, r2_ridge_train_pca, mae_ridge_train_pca, mse_ridge_test_pca, r2_ridge_test_pca, mae_ridge_test_pca))
print(f'Ridge Regression (PCA_Normalised) - Train MSE: {mse_ridge_train_pca}, Test MSE: {mse_ridge_test_pca}, Train R^2: {r2_ridge_train_pca}, Test R^2: {r2_ridge_test_pca}, Train MAE: {mae_ridge_train_pca}, Test MAE: {mae_ridge_test_pca}')

# Random Forest on PCA-transformed data
mse_rf_train_pca, r2_rf_train_pca, mae_rf_train_pca, mse_rf_test_pca, r2_rf_test_pca, mae_rf_test_pca = train_evaluate_model(RandomForestRegressor(), X_train_pca, y_train_pca, X_test_pca, y_test_pca)
results_2.append(('Random Forest', 'PCA_Normalised', mse_rf_train_pca, r2_rf_train_pca, mae_rf_train_pca, mse_rf_test_pca, r2_rf_test_pca, mae_rf_test_pca))
print(f'Random Forest (PCA_Normalised) - Train MSE: {mse_rf_train_pca}, Test MSE: {mse_rf_test_pca}, Train R^2: {r2_rf_train_pca}, Test R^2: {r2_rf_test_pca}, Train MAE: {mae_rf_train_pca}, Test MAE: {mae_rf_test_pca}')

# Feature engineering on the normalised data

# Converting the normalized dataset back to a DataFrame for easier handling
X_scaled_normalized_df = pd.DataFrame(X_scaled_normalized, columns=data_normalized.drop(columns=['Grade']).columns)

# Polynomial Feature Engineering

# Identify numeric features from the normalized data (excluding the target 'Grade')
numeric_features = X_scaled_normalized_df.columns

# Generate polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled_normalized_df[numeric_features])
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(numeric_features))

# Skewness Calculation

# Calculate the skewness for all numeric features in the normalized dataset
skewness = X_scaled_normalized_df.skew().sort_values(ascending=False)

# Display the skewness values
print("Skewness of Features in Normalized Data:")
print(skewness)

# Set a threshold for skewness to determine which features to log transform
threshold = 0.5

# Identify the features that have skewness greater than the threshold
skewed_features = skewness[(skewness > threshold) | (skewness < -threshold)].index

# Display the identified skewed features
print(f"Features with skewness greater than {threshold} or less than {-threshold}:")
print(skewed_features)

# Visualization of Skewed Features

# Visualize the distribution of the skewed features before log transformation
plt.figure(figsize=(14, len(skewed_features) * 4))

for i, feature in enumerate(skewed_features):
    plt.subplot(len(skewed_features), 1, i + 1)
    sns.histplot(X_scaled_normalized_df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature} (Skewness: {skewness[feature]:.2f})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Log Transformation of Skewed Features

# Apply log transformation to skewed features, adding a small constant to avoid log(0) or log(negative)
X_log_transformed_df = X_scaled_normalized_df.copy()
X_log_transformed_df[skewed_features] = X_log_transformed_df[skewed_features].apply(lambda x: np.log1p(x - x.min() + 1))


# Set up the figure and axes for subplots: distribution before and after the log transformation
num_features = len(skewed_features)
fig, axes = plt.subplots(num_features, 2, figsize=(14, 4 * num_features))

# Loop through each skewed feature and create the plots
for i, feature in enumerate(skewed_features):
    # Original feature distribution
    sns.histplot(X_scaled_normalized_df[feature], bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Original {feature} Distribution')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')
    
    # Log-transformed feature distribution
    sns.histplot(X_log_transformed_df[feature], bins=30, kde=True, ax=axes[i, 1])
    axes[i, 1].set_title(f'Log-Transformed {feature} Distribution')
    axes[i, 1].set_xlabel('Value')
    axes[i, 1].set_ylabel('Frequency')

# Adjust the layout for better readability
plt.tight_layout()
plt.show()

# Combine all engineered features with the original dataframe
X_combined_df = pd.concat([X_scaled_normalized_df, X_poly_df, X_log_transformed_df[skewed_features]], axis=1)

# Check for NaN values in the combined dataframe
nan_columns = X_combined_df.columns[X_combined_df.isna().any()].tolist()

print(f"Columns with NaN values: {nan_columns}")
print(X_combined_df[nan_columns].isna().sum())

# Split data into training and testing sets
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(X_combined_df, y_normalized, test_size=0.2, random_state=42)

# Linear Regression on combined features
mse_lr_train_comb, r2_lr_train_comb, mae_lr_train_comb, mse_lr_test_comb, r2_lr_test_comb, mae_lr_test_comb = train_evaluate_model(LinearRegression(), X_train_comb, y_train_comb, X_test_comb, y_test_comb)
results_2.append(('Linear Regression', 'Combined Features', mse_lr_train_comb, r2_lr_train_comb, mae_lr_train_comb, mse_lr_test_comb, r2_lr_test_comb, mae_lr_test_comb))
print(f'Linear Regression (Combined Features) - Train MSE: {mse_lr_train_comb}, Test MSE: {mse_lr_test_comb}, Train R^2: {r2_lr_train_comb}, Test R^2: {r2_lr_test_comb}, Train MAE: {mae_lr_train_comb}, Test MAE: {mae_lr_test_comb}')

# Lasso Regression on combined features
mse_lasso_train_comb, r2_lasso_train_comb, mae_lasso_train_comb, mse_lasso_test_comb, r2_lasso_test_comb, mae_lasso_test_comb = train_evaluate_model(Lasso(), X_train_comb, y_train_comb, X_test_comb, y_test_comb)
results_2.append(('Lasso Regression', 'Combined Features', mse_lasso_train_comb, r2_lasso_train_comb, mae_lasso_train_comb, mse_lasso_test_comb, r2_lasso_test_comb, mae_lasso_test_comb))
print(f'Lasso Regression (Combined Features) - Train MSE: {mse_lasso_train_comb}, Test MSE: {mse_lasso_test_comb}, Train R^2: {r2_lasso_train_comb}, Test R^2: {r2_lasso_test_comb}, Train MAE: {mae_lasso_train_comb}, Test MAE: {mae_lasso_test_comb}')

# Ridge Regression on combined features
mse_ridge_train_comb, r2_ridge_train_comb, mae_ridge_train_comb, mse_ridge_test_comb, r2_ridge_test_comb, mae_ridge_test_comb = train_evaluate_model(Ridge(), X_train_comb, y_train_comb, X_test_comb, y_test_comb)
results_2.append(('Ridge Regression', 'Combined Features', mse_ridge_train_comb, r2_ridge_train_comb, mae_ridge_train_comb, mse_ridge_test_comb, r2_ridge_test_comb, mae_ridge_test_comb))
print(f'Ridge Regression (Combined Features) - Train MSE: {mse_ridge_train_comb}, Test MSE: {mse_ridge_test_comb}, Train R^2: {r2_ridge_train_comb}, Test R^2: {r2_ridge_test_comb}, Train MAE: {mae_ridge_train_comb}, Test MAE: {mae_ridge_test_comb}')

# Random Forest on combined features 
rf_model_comb = RandomForestRegressor()
mse_rf_train_comb, r2_rf_train_comb, mae_rf_train_comb, mse_rf_test_comb, r2_rf_test_comb, mae_rf_test_comb = train_evaluate_model(rf_model_comb, X_train_comb, y_train_comb, X_test_comb, y_test_comb)
results_2.append(('Random Forest', 'Combined Features', mse_rf_train_comb, r2_rf_train_comb, mae_rf_train_comb, mse_rf_test_comb, r2_rf_test_comb, mae_rf_test_comb))
print(f'Random Forest (Combined Features) - Train MSE: {mse_rf_train_comb}, Test MSE: {mse_rf_test_comb}, Train R^2: {r2_rf_train_comb}, Test R^2: {r2_rf_test_comb}, Train MAE: {mae_rf_train_comb}, Test MAE: {mae_rf_test_comb}')

# Create a DataFrame from the results
results_df_1 = pd.DataFrame(results_2, columns=['Model', 'Data Type', 'Train MSE', 'Train R^2', 'Train MAE', 'Test MSE', 'Test R^2', 'Test MAE'])

# Specify the file name
file_name = 'Machine_learning_model_results_normalised_final.xlsx'

# Write the DataFrame to an Excel file 
results_df_1.to_excel(file_name, index=False)

print(f'Results have been saved to {file_name}')

# Visualisation of the performance of the models on the normalised dataset 

# Set the style for the plots
sns.set(style="whitegrid")

# Function to create line plots for comparison
def plot_line_comparison(df, metric, metric_name):
    plt.figure(figsize=(14, 8))
    
    # Melt the DataFrame to make it easier to plot
    df_melted = df.melt(id_vars=['Model', 'Data Type'], 
                        value_vars=[f'Train {metric}', f'Test {metric}'], 
                        var_name='Set', value_name=metric_name)
    
    # Create a line plot for each model with different data types
    sns.lineplot(x='Data Type', y=metric_name, hue='Model', style='Set', 
                 markers=True, dashes=False, data=df_melted, ci=None)
    
    plt.title(f'{metric_name} Comparison Across Models and Data Types')
    plt.ylabel(metric_name)
    plt.xlabel('Data Type')
    plt.xticks(rotation=45)
    plt.legend(title='Model and Set', loc='best')
    plt.tight_layout()
    plt.show()

# Plot MSE for all Data Types
plot_line_comparison(results_df_1, 'MSE', 'MSE')

# Plot R^2 for all Data Types
plot_line_comparison(results_df_1, 'R^2', 'R^2')

# Plot MAE for all Data Types
plot_line_comparison(results_df_1, 'MAE', 'MAE')

# Hyperparameter tuning 

# Import library for tuning the models
from sklearn.model_selection import GridSearchCV

# Define function for tuning models
def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

# Define function for evaluating models (including both training and testing errors)
def evaluate_best_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    # Predictions on training set
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Predictions on testing set
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    return train_mse, train_r2, train_mae, test_mse, test_r2, test_mae

# Initialize a list to store all results
all_results = []

# Define models and their parameter grids
model_param_grids = {
    'Linear Regression': (LinearRegression(), {
        'fit_intercept': [True],
        'copy_X': [True],
        'positive': [False]
    }),
    'Lasso': (Lasso(), {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [5000]
    }),
    'Ridge': (Ridge(), {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'max_iter': [5000]
    }),
    'Random Forest': (RandomForestRegressor(), {
        'n_estimators': [100, 300],    
        'max_depth': [10, 20],
        'min_samples_split': [5,10],
        'min_samples_leaf': [2,4],
        'bootstrap': [True]
    })
}

# List of datasets to include in the tuning and evaluation
datasets = {
    'Normalized': (X_train_norm, X_test_norm, y_train_norm, y_test_norm),
    'PCA Transformed': (X_train_pca, X_test_pca, y_train_pca, y_test_pca),
    'Feature Engineered': (X_train_comb, X_test_comb, y_train_comb, y_test_comb)
}

# Function to handle hyperparameter tuning and evaluation for each dataset
def tune_and_evaluate_for_dataset(data_name, X_train_set, X_test_set, y_train_set, y_test_set):
    print(f"\nProcessing {data_name}...")

    # Process each model one by one
    for model_name, (model, param_grid) in model_param_grids.items():
        print(f"  Tuning {model_name} on {data_name}...")

        # Hyperparameter tuning
        best_params, _ = tune_model(model, param_grid, X_train_set, y_train_set)

        # Re-initialize the model with the best parameters
        model.set_params(**best_params)

        # Evaluate the tuned model and calculate both training and testing errors
        train_mse, train_r2, train_mae, test_mse, test_r2, test_mae = evaluate_best_model(
            model, X_train_set, y_train_set, X_test_set, y_test_set
        )

        # Store the results for this model
        all_results.append((
            data_name, model_name, best_params, 
            train_mse, train_r2, train_mae, 
            test_mse, test_r2, test_mae
        ))

# Loop over each dataset and process them
for data_name, (X_train_set, X_test_set, y_train_set, y_test_set) in datasets.items():
    tune_and_evaluate_for_dataset(data_name, X_train_set, X_test_set, y_train_set, y_test_set)

# Convert the results into a DataFrame
results_df_2 = pd.DataFrame(all_results, columns=[
    'Data Type', 'Model', 'Best Parameters', 
    'Train MSE', 'Train R^2', 'Train MAE', 
    'Test MSE', 'Test R^2', 'Test MAE'
])

# Save the results to an Excel file
output_file = 'hyperparameter_tuning_results_final.xlsx'
results_df_2.to_excel(output_file, index=False)

# Print confirmation
print(f'\nHyperparameter tuning and evaluation results have been saved to {output_file}')

# Visualisation of the performance of machine learning model

# Calculate benchmark values for all metrics (assuming these are pre-calculated)
simple_avg_mse = 2.2896530602940173
simple_avg_r2 = 0.6486128487763867
simple_avg_mae = 1.0107717033098549

# Construct the benchmark data to have the correct lengths
benchmark_data = {
    'Model': ['Benchmark'] * 3,
    'Data Type': ['Normalized', 'PCA Transformed', 'Feature Engineered'],
    'Train MSE': [simple_avg_mse] * 3,
    'Train R^2': [simple_avg_r2] * 3,
    'Train MAE': [simple_avg_mae] * 3,
    'Test MSE': [simple_avg_mse] * 3,
    'Test R^2': [simple_avg_r2] * 3,
    'Test MAE': [simple_avg_mae] * 3
}

# Convert the dictionary to a DataFrame
benchmark_df = pd.DataFrame(benchmark_data)

# Append the benchmark DataFrame to the results DataFrame
results_df_2 = pd.concat([results_df_2, benchmark_df], ignore_index=True)

# Function to create point plots for train and test metrics for all models with benchmark
def plot_train_test_performance_pointplot(df, benchmark_values):
    plt.figure(figsize=(18, 8))
    
    # Melt the DataFrame to make it easier to plot for Train metrics
    df_melted_train = df.melt(id_vars=['Model', 'Data Type'], 
                              value_vars=['Train MSE', 'Train R^2', 'Train MAE'], 
                              var_name='Metric', value_name='Value')
    
    # Create the point plot for Train metrics
    plt.subplot(1, 2, 1)
    sns.pointplot(x='Model', y='Value', hue='Metric', data=df_melted_train, markers=["o", "s", "D"], linestyles=["-", "--", "-."])
    
    # Plot benchmark points for Train
    plt.axhline(y=benchmark_values['MSE'], color='red', linestyle='--', label='Benchmark MSE')
    plt.axhline(y=benchmark_values['R^2'], color='green', linestyle='--', label='Benchmark R^2')
    plt.axhline(y=benchmark_values['MAE'], color='blue', linestyle='--', label='Benchmark MAE')
    
    plt.title('Train Performance Across Models (MSE, R^2, MAE)')
    plt.ylabel('Metric Value')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(title='Metric', loc='best')
    
    # Melt the DataFrame to make it easier to plot for Test metrics
    df_melted_test = df.melt(id_vars=['Model', 'Data Type'], 
                             value_vars=['Test MSE', 'Test R^2', 'Test MAE'], 
                             var_name='Metric', value_name='Value')
    
    # Create the point plot for Test metrics
    plt.subplot(1, 2, 2)
    sns.pointplot(x='Model', y='Value', hue='Metric', data=df_melted_test, markers=["o", "s", "D"], linestyles=["-", "--", "-."])
    
    # Plot benchmark points for Test
    plt.axhline(y=benchmark_values['MSE'], color='red', linestyle='--', label='Benchmark MSE')
    plt.axhline(y=benchmark_values['R^2'], color='green', linestyle='--', label='Benchmark R^2')
    plt.axhline(y=benchmark_values['MAE'], color='blue', linestyle='--', label='Benchmark MAE')
    
    plt.title('Test Performance Across Models (MSE, R^2, MAE)')
    plt.ylabel('Metric Value')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(title='Metric', loc='best')
    
    plt.tight_layout()
    plt.show()

# Generate the point plots for Train and Test metrics for all models with benchmark
plot_train_test_performance_pointplot(results_df_2, {
    'MSE': simple_avg_mse,
    'R^2': simple_avg_r2,
    'MAE': simple_avg_mae
})

# Data for the bar chart
metrics = ['MSE', 'R^2', 'MAE']
values = [simple_avg_mse, simple_avg_r2, simple_avg_mae]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['red', 'green', 'blue'])

# Add titles and labels
plt.title('Benchmark Model Performance')
plt.ylabel('Metric Value')

# Display the value on top of each bar
for i, value in enumerate(values):
    plt.text(i, value + 0.05, f'{value:.3f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()


# # Extra graphs 

# # Data for the benchmark and epochs
# benchmark_data = {
#     'Metric': ['MSE', 'R^2', 'MAE'],
#     'Benchmark': [3.1677707006369427, 0.3023928811935258, 1.4105095541401274]
# }

# epochs_data = {
#     'Epocs': [50, 100],
#     'MSE': [2.0, 1.84],
#     'R^2': [0.7, 0.72],
#     'MAE': [0.98, 0.93]
# }

# # Convert to DataFrame
# benchmark_df = pd.DataFrame(benchmark_data)
# epochs_df = pd.DataFrame(epochs_data)

# # Plot the bar chart
# plt.figure(figsize=(12, 6))

# # Plot benchmark values
# plt.bar(benchmark_df['Metric'], benchmark_df['Benchmark'], color='grey', alpha=0.7, label='Benchmark')

# # Plot values for each epoch
# width = 0.2  # width of the bars
# plt.bar([x - width for x in range(len(benchmark_df['Metric']))], epochs_df.loc[0, ['MSE', 'R^2', 'MAE']], 
#         width=width, color='blue', alpha=0.7, label='Epoch 50')
# plt.bar([x for x in range(len(benchmark_df['Metric']))], epochs_df.loc[1, ['MSE', 'R^2', 'MAE']], 
#         width=width, color='orange', alpha=0.7, label='Epoch 100')

# # Adding labels and titles
# plt.title('Model Performance Comparison with Benchmark')
# plt.ylabel('Metric Value')
# plt.xticks(range(len(benchmark_df['Metric'])), benchmark_df['Metric'])
# plt.legend()

# # Show values on top of bars
# for i, value in enumerate(benchmark_df['Benchmark']):
#     plt.text(i, value + 0.05, f'{value:.2f}', ha='center', fontsize=10)
    
# for i, value in enumerate(epochs_df.loc[0, ['MSE', 'R^2', 'MAE']]):
#     plt.text(i - width, value + 0.05, f'{value:.2f}', ha='center', fontsize=10, color='blue')

# for i, value in enumerate(epochs_df.loc[1, ['MSE', 'R^2', 'MAE']]):
#     plt.text(i, value + 0.05, f'{value:.2f}', ha='center', fontsize=10, color='orange')

# plt.tight_layout()
# plt.show()

# # New data including both training and test metrics for each epoch
# epochs_data_extended = {
#     'Epocs': ['Epoch 50 Train', 'Epoch 50 Test', 'Epoch 100 Train', 'Epoch 100 Test'],
#     'MSE': [1.8, 1.9, 1.85, 1.93],
#     'R^2': [0.68, 0.66, 0.69, 0.65],
#     'MAE': [0.95, 1.02, 0.96, 1.93]
# }

# # Benchmark values (as previously defined)
# benchmark_data = {
#     'Metric': ['MSE', 'R^2', 'MAE'],
#     'Benchmark': [3.1677707006369427, 0.3023928811935258, 1.4105095541401274]
# }

# # Convert to DataFrame
# benchmark_df = pd.DataFrame(benchmark_data)
# epochs_df_extended = pd.DataFrame(epochs_data_extended)

# # Plot the bar chart with larger labels and pastel colors
# plt.figure(figsize=(14, 8))

# # Plot benchmark values with increased width and pastel color
# plt.bar(benchmark_df['Metric'], benchmark_df['Benchmark'], color='grey', alpha=0.7, width=0.25, label='Benchmark')

# # Plot values for each training and test epoch with increased width and pastel colors
# width = 0.25  # Adjust width for more bars
# positions = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
# colors = ['#FFB6C1', '#87CEFA', '#FFDAB9', '#98FB98']  # Pastel colors

# for i, epoch in enumerate(epochs_df_extended['Epocs']):
#     plt.bar([x + positions[i % 4] for x in range(len(benchmark_df['Metric']))], 
#             epochs_df_extended.iloc[i, 1:],
#             width=width, alpha=0.7, label=epoch, color=colors[i])

# # Adding labels and titles with increased font size
# plt.title('Model Performance Comparison: Training, Test, and Benchmark', fontsize=20)
# plt.ylabel('Metric Value', fontsize=18)
# plt.xticks(range(len(benchmark_df['Metric'])), benchmark_df['Metric'], fontsize=18)
# plt.legend(fontsize=14)

# # Show values on top of bars with larger font size
# for i, value in enumerate(benchmark_df['Benchmark']):
#     plt.text(i, value + 0.05, f'{value:.2f}', ha='center', fontsize=16)
    
# for i in range(len(epochs_df_extended)):
#     for j in range(len(benchmark_df['Metric'])):
#         plt.text(j + positions[i % 4], epochs_df_extended.iloc[i, j+1] + 0.05, 
#                   f'{epochs_df_extended.iloc[i, j+1]:.2f}', 
#                   ha='center', fontsize=16, color='black')

# plt.tight_layout()
# plt.show()

# # Data for the model after hyperparameter tuning
# tuned_train_data = {
#     'Epocs': ['Epoch 50 Train', 'Epoch 100 Train'],
#     'MSE': [1.78, 1.82],
#     'R^2': [0.74, 0.73],
#     'MAE': [0.92, 0.94]
# }

# tuned_test_data = {
#     'Epocs': ['Epoch 50 Test', 'Epoch 100 Test'],
#     'MSE': [1.56, 1.66],
#     'R^2': [0.72, 0.70],
#     'MAE': [0.89, 0.90]
# }

# # Data for the model without hyperparameter tuning
# untuned_data = {
#     'Epocs': ['Epoch 50 Untuned', 'Epoch 100 Untuned'],
#     'MSE': [1.56, 1.66],
#     'R^2': [0.72, 0.70],
#     'MAE': [0.89, 0.90]
# }

# # Convert to DataFrames
# benchmark_df = pd.DataFrame(benchmark_data)
# tuned_train_df = pd.DataFrame(tuned_train_data)
# tuned_test_df = pd.DataFrame(tuned_test_data)
# untuned_df = pd.DataFrame(untuned_data)

# # Setting up data for the grouped bar chart
# metrics = ['MSE', 'R^2', 'MAE']
# values = np.array([tuned_train_df.iloc[:, 1:].values,
#                     tuned_test_df.iloc[:, 1:].values,
#                     untuned_df.iloc[:, 1:].values])

# # Plotting the grouped bar chart
# fig, ax = plt.subplots(figsize=(14, 8))

# # Set positions for the groups of bars
# bar_width = 0.2
# bar_positions = np.arange(len(metrics))

# # Plotting each condition
# colors = ['#FFB6C1', '#87CEFA', '#FFDAB9', '#98FB98']  # Pastel colors
# for i, key in enumerate(tuned_train_df['Epocs']):
#     ax.bar(bar_positions + i * bar_width, tuned_train_df.iloc[i, 1:], width=bar_width, label=key, color=colors[i])

# for i, key in enumerate(tuned_test_df['Epocs']):
#     ax.bar(bar_positions + (i + 2) * bar_width, tuned_test_df.iloc[i, 1:], width=bar_width, label=key, color=colors[i + 2])

# for i, key in enumerate(untuned_df['Epocs']):
#     ax.bar(bar_positions + (i + 4) * bar_width, untuned_df.iloc[i, 1:], width=bar_width, label=key, color='purple')

# # Plot benchmark values
# ax.bar(bar_positions + 6 * bar_width, benchmark_df['Benchmark'], width=bar_width, color='darkgrey', alpha=0.7, label='Benchmark')

# # Adding labels and title with updated font size and weight
# ax.set_xlabel('Metrics', fontsize=16, color='black')
# ax.set_ylabel('Values', fontsize=16, color='black')
# ax.set_title('Model Performance Comparison: Before and After Hyperparameter Tuning', fontsize=18, color='black')
# ax.set_xticks(bar_positions + (6 / 2) * bar_width)
# ax.set_xticklabels(metrics, fontsize=14, color='black')
# ax.legend(fontsize=12)

# # Show values on top of bars with larger font size and bold
# for i in range(len(tuned_train_df)):
#     for j in range(len(metrics)):
#         ax.text(bar_positions[j] + i * bar_width, tuned_train_df.iloc[i, j+1] + 0.03, 
#                 f'{tuned_train_df.iloc[i, j+1]:.2f}', ha='center', fontsize=14, fontweight='bold', color='black')

# for i in range(len(tuned_test_df)):
#     for j in range(len(metrics)):
#         ax.text(bar_positions[j] + (i + 2) * bar_width, tuned_test_df.iloc[i, j+1] + 0.03, 
#                 f'{tuned_test_df.iloc[i, j+1]:.2f}', ha='center', fontsize=14, fontweight='bold', color='black')

# for i in range(len(untuned_df)):
#     for j in range(len(metrics)):
#         ax.text(bar_positions[j] + (i + 4) * bar_width, untuned_df.iloc[i, j+1] + 0.03, 
#                 f'{untuned_df.iloc[i, j+1]:.2f}', ha='center', fontsize=14, fontweight='bold', color='black')

# # Benchmark values
# for j in range(len(metrics)):
#     ax.text(bar_positions[j] + 6 * bar_width, benchmark_df['Benchmark'][j] + 0.03, 
#             f'{benchmark_df["Benchmark"][j]:.2f}', ha='center', fontsize=14, fontweight='bold', color='black')

# plt.tight_layout()
# plt.show()





# # Data from the table provided
# data_table = {
#     'Model': [
#         'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Random Forest',
#         'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Random Forest',
#         'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Random Forest'
#     ],
#     'Data Type': [
#         'Normalized', 'Normalized', 'Normalized', 'Normalized',
#         'PCA_Normalised', 'PCA_Normalised', 'PCA_Normalised', 'PCA_Normalised',
#         'Combined Features', 'Combined Features', 'Combined Features', 'Combined Features'
#     ],
#     'Train MSE': [
#         2.356475151, 3.83886804, 2.373523792, 0.414065694,
#         2.416807003, 2.716267444, 2.416812262, 0.433362912,
#         1.2548E-27, 3.83886804, 0.590478534, 0.429905711
#     ],
#     'Train R^2': [
#         0.474450711, 0.143842289, 0.470648463, 0.907653628,
#         0.460995291, 0.394208581, 0.460994118, 0.903349895,
#         1, 0.143842289, 0.868309422, 0.904120932
#     ],
#     'Train MAE': [
#         1.226343296, 1.609198301, 1.218164454, 0.517263119,
#         1.230191104, 1.335347507, 1.230359453, 0.528973061,
#         2.63291E-14, 1.609198301, 0.584380633, 0.531731187
#     ]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data_table)

# # Colors to use
# colors = ['#FFB6C1', '#87CEFA', '#FFDAB9', '#98FB98', '#DDA0DD', '#FFA07A', '#20B2AA', '#778899', '#B0E0E6', '#40E0D0', '#FF6347', '#FAFAD2']

# # Create Lollipop chart for Train Data
# metrics_train = ['Train MSE', 'Train R^2', 'Train MAE']
# x_train = np.arange(len(metrics_train))
# bar_width = 0.15

# fig, ax = plt.subplots(figsize=(14, 8))

# # Plot for Train Data with specified line thickness and color
# for i, (model, dtype) in enumerate(zip(df['Model'], df['Data Type'])):
#     markerline, stemline, baseline = ax.stem(x_train + i * bar_width, df.iloc[i][metrics_train], linefmt='-', markerfmt='o', basefmt=" ")
#     plt.setp(stemline, color='yellow', linewidth=2)  # Thicker yellow stem
#     plt.setp(markerline, color=colors[i], markersize=10)  # Marker color as per the list

# ax.set_xlabel('Metrics', fontsize=12)
# ax.set_ylabel('Values', fontsize=12)
# ax.set_title('Train Performance Comparison Across Models and Data Types', fontsize=14)
# ax.set_xticks(x_train + len(df) * bar_width / len(metrics_train))
# ax.set_xticklabels(metrics_train, fontsize=12)
# ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1,1))

# # Data from the table provided
# data_90 = {
#     'Model': [
#         'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Random Forest',
#         'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Random Forest',
#         'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Random Forest'
#     ],
#     'Data Type': [
#         'Normalized', 'Normalized', 'Normalized', 'Normalized',
#         'PCA_Normalised', 'PCA_Normalised', 'PCA_Normalised', 'PCA_Normalised',
#         'Combined Features', 'Combined Features', 'Combined Features', 'Combined Features'
#     ],
#     'Test MSE': [
#         3.738037252, 4.126315979, 3.841726181, 4.098917785,
#         3.8469101, 3.882314116, 3.847498717, 3.871634997,
#         46.38182246, 4.126315979, 8.227882187, 4.237413672
#     ],
#     'Test R^2': [
#         0.107495757, 0.014789241, 0.082738698, 0.02133091,
#         0.081500971, 0.073047809, 0.081360431, 0.075597585,
#         -10.07425383, 0.014789241, -0.964512194, -0.011736756
#     ],
#     'Test MAE': [
#         1.523204982, 1.596152513, 1.560579355, 1.627064055,
#         1.567569485, 1.578796722, 1.567622152, 1.588197028,
#         5.356497497, 1.596152513, 2.171554035, 1.647990514
#     ]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data_90)

# # Colors to use
# colors = ['#FFB6C1', '#87CEFA', '#FFDAB9', '#98FB98', '#DDA0DD', '#FFA07A', '#20B2AA', '#778899', '#B0E0E6', '#40E0D0', '#FF6347', '#FAFAD2']

# # Create Lollipop chart for Test Data
# metrics_test = ['Test MSE', 'Test R^2', 'Test MAE']
# x_test = np.arange(len(metrics_test))
# bar_width = 0.15

# fig, ax = plt.subplots(figsize=(14, 8))

# # Plot for Test Data with specified line thickness and color
# for i, (model, dtype) in enumerate(zip(df['Model'], df['Data Type'])):
#     markerline, stemline, baseline = ax.stem(x_test + i * bar_width, df.iloc[i][metrics_test], linefmt='-', markerfmt='o', basefmt=" ")
#     plt.setp(stemline, color='yellow', linewidth=2)  # Thicker yellow stem
#     plt.setp(markerline, color=colors[i], markersize=10)  # Marker color as per the list

# ax.set_xlabel('Metrics', fontsize=12)
# ax.set_ylabel('Values', fontsize=12)
# ax.set_title('Test Performance Comparison Across Models and Data Types', fontsize=14)
# ax.set_xticks(x_test + len(df) * bar_width / len(metrics_test))
# ax.set_xticklabels(metrics_test, fontsize=12)
# ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1,1))

# plt.tight_layout()
# plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:55:38 2024

@author: 5558022_neural_network

To be used in a seperate script

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
data = pd.read_excel('PredictionsData.xlsx')
data.copy = data

# Normalisation of the COVID variable
# Assuming 'COVID' is the column or a flag that indicates COVID-related rows
covid_data = data[data['COVID'] == 1]  # Filter rows related to COVID
non_covid_data = data[data['COVID'] == 0]  # Filter rows not related to COVID

# Normalize the COVID-related data (MinMax scaling in this example)
scaler_covid = MinMaxScaler()
covid_data_scaled = covid_data.copy()
covid_data_scaled.iloc[:, :-1] = scaler_covid.fit_transform(covid_data.iloc[:, :-1])  # Apply scaling to all features except the target

# Merge the data back together
data_normalized = pd.concat([covid_data_scaled, non_covid_data])

# Standardize the features for the whole dataset (including the normalized COVID data)
scaler = StandardScaler()
X = data_normalized.drop('Grade', axis=1)
y = data_normalized['Grade']
X_scaled = scaler.fit_transform(X)

# Eastablishing the benchmark 

# Define the columns for predicted Maths, English grades, and the target Engineering grade
predicted_math_col = 'Metric27'
predicted_english_col = 'Metric12'   
engineering_grade_col = 'Grade' 

# Benchmark Calculation - Simple Average of Maths and English
data['Simple_Average'] = data[[predicted_math_col, predicted_english_col]].mean(axis=1)
simple_avg_mse = mean_squared_error(data[engineering_grade_col], data['Simple_Average'])
simple_avg_r2 = r2_score(data[engineering_grade_col], data['Simple_Average'])
simple_avg_mae = mean_absolute_error(data[engineering_grade_col], data['Simple_Average'])

print(f'Benchmark - MSE: {simple_avg_mse}, R^2: {simple_avg_r2}, MAE: {simple_avg_mae}')

# Drop 'Simple_Average' from the data before further processing
data_normalized = data_normalized.drop(columns=['Simple_Average'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Display the model's architecture
model.summary()

# Train the neural network model on 50 epocs
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# # Train the neural network model on 100 epocs
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test data
loss, mse = model.evaluate(X_test, y_test, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R^2 score and MAE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Calculate training error metrics
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

print(f'Training Mean Squared Error: {mse_train}')
print(f'Training R^2 Score: {r2_train}')
print(f'Training Mean Absolute Error: {mae_train}')

# Further Neural Network with Grid Search

# Define the neural network architecture as a function
def create_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Custom Keras Regressor class to use in GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, optimizer='adam', dropout_rate=0.5, epochs=50, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y):
        self.model_ = self.build_fn(optimizer=self.optimizer, dropout_rate=self.dropout_rate)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return self.model_.predict(X)

# Create the Keras regressor
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the grid search parameters
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64],
    'epochs': [50, 100],
    'dropout_rate': [0.3, 0.5, 0.7]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Get the best model
best_model = grid_result.best_estimator_

# Evaluate the model on the test data
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Evaluate the model on the training data
y_pred_train = best_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

print(f'Training Mean Squared Error (After Hyperparameter Tuning): {mse_train}')
print(f'Training R^2 Score (After Hyperparameter Tuning): {r2_train}')
print(f'Training Mean Absolute Error (After Hyperparameter Tuning): {mae_train}')


# Plot training & validation loss values - Performance graph
plt.figure(figsize=(10, 6))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Neural Network Performance - Loss Over 100 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend(loc='upper right')
plt.grid(True)

plt.show()

# Neural network comparison with the benchmark

benchmark_mse = simple_avg_mse
benchmark_r2 = simple_avg_r2
benchmark_mae = simple_avg_mae

initial_mse = mse
initial_r2 = r2
initial_mae = mae

tuned_test_mse = mse  # After tuning, on the test set
tuned_test_r2 = r2    # After tuning, on the test set
tuned_test_mae = mae  # After tuning, on the test set

tuned_train_mse = mse_train  # After tuning, on the training set
tuned_train_r2 = r2_train    # After tuning, on the training set
tuned_train_mae = mae_train  # After tuning, on the training set

# Metrics to plot
metrics = ['MSE', 'R^2', 'MAE']

# Combine results
benchmark_values = [benchmark_mse, benchmark_r2, benchmark_mae]
initial_values = [initial_mse, initial_r2, initial_mae]
tuned_test_values = [tuned_test_mse, tuned_test_r2, tuned_test_mae]
tuned_train_values = [tuned_train_mse, tuned_train_r2, tuned_train_mae]

# Set up the bar chart
x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - 1.5*width, benchmark_values, width, label='Benchmark')
rects2 = ax.bar(x - 0.5*width, initial_values, width, label='Initial Predictions')
rects3 = ax.bar(x + 0.5*width, tuned_test_values, width, label='Tuned Predictions (Test Set)')
rects4 = ax.bar(x + 1.5*width, tuned_train_values, width, label='Tuned Predictions (Training Set)')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_title('Comparison of Benchmark, Initial, and Tuned Predictions with Training Error')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Attach a text label above each bar in *rects*, displaying its height.
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()

# # Save the final tuned model - this is to save the model to run for new excel with predicted scores.
# best_model.model_.save('trained_model.h5')




















