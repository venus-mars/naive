# Step 1: Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Step 2: Data Collection
# Load the dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Step 3: Exploratory Data Analysis (EDA) and Feature Engineering
# Checking for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Handle missing values (optional) - fill missing values in 'Distance_from_Home' with the most frequent value
df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0], inplace=True)

# Display the first few rows to understand the dataset
print("First few rows of the dataset:")
print(df.head())

# Feature Engineering: If necessary, encode categorical features using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Checking correlations (optional) if the dataset is numerical
correlation_matrix = df_encoded.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Step 4: Creating Frequency Tables
# Splitting the dataset into features (X) and the target (y)
X = df_encoded.drop('Exam_Score', axis=1)  # Drop the target variable from the features
y = df_encoded['Exam_Score']  # Set the target variable

# Display unique classes in the target column
print("Target Classes (Exam Scores):", np.unique(y))

# Creating a frequency table for each feature
for column in X.columns:
    print(f"Frequency Table for {column}:")
    print(X[column].value_counts())
    print()

# Step 5: Calculating Prior Probabilities and Likelihoods
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Calculate prior probabilities (P(A)) and likelihoods (P(B|A)) using the training set
class_priors = model.class_prior_  # Prior probabilities for each class
print("Class Priors (P(A)):")
print(class_priors)

# Likelihood for each feature given the class (P(B|A)) can be obtained from the model
print("Feature Likelihoods (P(B|A)):")
print(model.theta_)  # Mean of each feature per class
print(model.sigma_)  # Variance of each feature per class

# Step 6: Applying Bayes' Theorem for Prediction
# Make predictions on the test set
y_pred = model.predict(X_test)

# Display the predictions
print("Predictions on test data:")
print(y_pred)

# Evaluate the model using accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Probability of a specific test case using Bayes' Theorem manually
# For demonstration, let’s calculate P(Yes|features) for a sample test case
sample = X_test.iloc[0]  # Taking the first test case
probabilities = model.predict_proba([sample])

print("Probabilities for the sample test case:")
print(probabilities)
