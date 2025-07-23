import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA # Optional, for transformation example

# --- 1. Define a Sample Dataset ---
# Let's create a synthetic dataset resembling customer data for a churn prediction model.
data = {
    'CustomerID': range(1, 101),
    'Age': [25, 30, 45, 22, None, 35, 28, 50, 40, 33] * 10,
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female'] * 10,
    'MonthlyCharges': [50.0, 75.5, 90.0, 30.0, 60.0, 80.0, None, 45.0, 100.0, 70.0] * 10,
    'TotalCharges': [1200.0, 3500.0, 8000.0, 500.0, None, 4000.0, 2000.0, 6000.0, 9500.0, 3000.0] * 10,
    'Contract': ['Month-to-month', 'Two year', 'One year', 'Month-to-month', 'Month-to-month', 'Two year', 'One year', 'Two year', 'Month-to-month', 'One year'] * 10,
    'HasInternetService': [True, False, True, True, False, True, True, False, True, True] * 10,
    'Churn': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 10 # Target variable
}
df = pd.DataFrame(data)

# Introduce some more missing values to demonstrate imputation
import numpy as np
df.loc[df.sample(frac=0.05).index, 'MonthlyCharges'] = np.nan
df.loc[df.sample(frac=0.03).index, 'Age'] = np.nan
df.loc[df.sample(frac=0.02).index, 'TotalCharges'] = np.nan
df.loc[df.sample(frac=0.01).index, 'Gender'] = np.nan

print("Original DataFrame Head:")
print(df.head())
print("\nMissing Values before Preprocessing:")
print(df.isnull().sum())
print("-" * 50)

# Separate features (X) and target (y)
X = df.drop(['CustomerID', 'Churn'], axis=1) # CustomerID is just an identifier, Churn is the target
y = df['Churn']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
boolean_features = X.select_dtypes(include='bool').columns.tolist() # Boolean features can be treated as numerical or encoded

# For simplicity, let's treat boolean features as numerical for now or specifically handle them.
# In this case, boolean features are already in a numerical-like format (True/False)
# If we wanted to encode them, OneHotEncoder would work, but typically they are fine as is for many models.
# For demonstration, we'll treat them as numerical, and StandardScaler will handle them.
numerical_features.extend(boolean_features)


# --- 2. Data Preprocessing & Transformation Pipeline ---

# Create preprocessing steps for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute missing numerical values with the mean
    ('scaler', StandardScaler())                # Scale numerical features
])

# Create preprocessing steps for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with the most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))   # One-hot encode categorical features
])

# Create a ColumnTransformer to apply different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) as they are
)

# Create the full data processing pipeline
# We can add more steps here, like dimensionality reduction (PCA) or custom transformers
data_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('pca', PCA(n_components=0.95)) # Optional: Add PCA for dimensionality reduction, keeping 95% variance
])

# --- 3. Apply the Pipeline (Fit and Transform) ---

# It's good practice to split data into training and testing sets *before* fitting the pipeline
# to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data and transform both training and testing data
X_train_processed = data_pipeline.fit_transform(X_train)
X_test_processed = data_pipeline.transform(X_test)

print("\nShape of X_train_processed:", X_train_processed.shape)
print("Shape of X_test_processed:", X_test_processed.shape)

# To inspect the transformed data, especially after OneHotEncoder, it's useful to get feature names.
# This requires a bit more effort due to ColumnTransformer and OneHotEncoder's dynamic output.

# Get feature names after preprocessing
def get_feature_names(column_transformer):
    output_features = []
    for name, preprocessor, features in column_transformer.transformers_:
        if hasattr(preprocessor, 'get_feature_names_out'): # For OneHotEncoder
            if isinstance(features, str): # Handle case where features is a string (e.g., 'remainder')
                output_features.extend([features])
            else:
                output_features.extend(preprocessor.get_feature_names_out(features))
        elif isinstance(features, list): # For numerical features
            output_features.extend(features)
        else: # For other cases, just add the name (e.g., 'remainder' columns)
            output_features.append(name)
    return output_features

# If PCA was included, the feature names would change to 'pca0', 'pca1', etc.
# For now, let's assume no PCA to make feature name inspection easier.
# If PCA is active, X_train_processed would be an array of PCA components.

# Get feature names from the preprocessor (before optional PCA)
preprocessor_feature_names = data_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Create a DataFrame from the processed training data (for inspection)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=preprocessor_feature_names)

print("\nProcessed Training Data Head (first 5 rows and selected columns for brevity):")
print(X_train_processed_df.head())
print("\nDescriptive Statistics of Processed Data (first few columns):")
print(X_train_processed_df.iloc[:, :5].describe()) # Show describe for first 5 columns

print("\n--- ETL Process Completed ---")

# --- 4. Data Loading (Simulation) ---

# The processed data (X_train_processed, X_test_processed, y_train, y_test) is now ready.
# You can now:

# A. Save to CSV files
# Ensure the processed data is in a DataFrame if you want column headers
processed_data_all = data_pipeline.fit_transform(X) # Fit and transform on full dataset for final output
final_feature_names = data_pipeline.named_steps['preprocessor'].get_feature_names_out()
processed_df_final = pd.DataFrame(processed_data_all, columns=final_feature_names)
processed_df_final['Churn'] = y.values # Add the target variable back

output_filename_csv = 'processed_data.csv'
processed_df_final.to_csv(output_filename_csv, index=False)
print(f"\n1. Data saved to '{output_filename_csv}' (CSV format).")
print(f"Shape of final processed data saved to CSV: {processed_df_final.shape}")


# B. Ready for Machine Learning Model Training
print("\n2. Processed data is ready for Machine Learning Model training:")
print(f"   X_train_processed (features for training): shape {X_train_processed.shape}")
print(f"   y_train (target for training): shape {y_train.shape}")
print(f"   X_test_processed (features for testing): shape {X_test_processed.shape}")
print(f"   y_test (target for testing): shape {y_test.shape}")

# Example: Train a simple Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42, solver='liblinear') # Using liblinear for robustness
model.fit(X_train_processed, y_train)
accuracy = model.score(X_test_processed, y_test)
print(f"\n3. Example: A Logistic Regression model trained on the processed data achieved an accuracy of: {accuracy:.4f}")

# C. Save the trained pipeline for future use (e.g., deploying for new predictions)
import joblib
pipeline_filename = 'data_preprocessing_pipeline.joblib'
joblib.dump(data_pipeline, pipeline_filename)
print(f"\n4. Data preprocessing pipeline saved to '{pipeline_filename}'.")
print("You can load this pipeline later to preprocess new, unseen data using: `loaded_pipeline = joblib.load('data_preprocessing_pipeline.joblib')`")
print("Then, `new_processed_data = loaded_pipeline.transform(new_raw_data)`.")