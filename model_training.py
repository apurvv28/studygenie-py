import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load the dataset from CSV
df = pd.read_csv('test.csv')
print("Dataset loaded successfully.")

# Preprocess the data
print("Preprocessing data...")
X = df.drop("learning_style", axis=1)
y = df["learning_style"]

# Encode the responses
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le
print("Data preprocessing completed.")

# Encode the target variable
print("Encoding target variable...")
le_target = LabelEncoder()
y = le_target.fit_transform(y)
print("Target variable encoded.")

# Split the data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split completed.")

# Hyperparameter tuning using Grid Search
print("Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Train the model with the best parameters
print("Training the model with the best parameters...")
best_model.fit(X_train, y_train)
print("Model training completed successfully.")

# Evaluate the model
print("Evaluating the model...")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and encoders
print("Saving the model and encoders...")
with open('models/study_style_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('models/label_encoders.pkl', 'wb') as encoders_file:
    pickle.dump(label_encoders, encoders_file)

with open('models/target_encoder.pkl', 'wb') as target_encoder_file:
    pickle.dump(le_target, target_encoder_file)
print("Model and encoders saved successfully.")