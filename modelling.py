# import mlflow
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import random
# import numpy as np

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# # Create a new MLflow Experiment
# mlflow.set_experiment("Latihan Credit Scoring")

# data = pd.read_csv("train_pca.csv")

# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop("Credit_Score", axis=1),
#     data["Credit_Score"],
#     random_state=42,
#     test_size=0.2
# )
# input_example = X_train[0:5]

# with mlflow.start_run():
#     # Log parameters
#     n_estimators = 505
#     max_depth = 37
#     mlflow.autolog()
#     # Train model
#     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#     mlflow.sklearn.log_model(
#         sk_model=model,
#         name="model",
#         input_example=input_example
#     )
#     model.fit(X_train, y_train)
#     # Log metrics
#     accuracy = model.score(X_test, y_test)
#     mlflow.log_metric("accuracy", accuracy)

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set experiment name
mlflow.set_experiment("Latihan Credit Scoring")

# Load dataset
data = pd.read_csv("train_pca.csv")

# Split data into features and target
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)

# Input example for model logging
input_example = X_train.iloc[0:5]

with mlflow.start_run():
    # Automatically log sklearn params, metrics, and model
    mlflow.autolog()

    # Define model hyperparameters
    n_estimators = 505
    max_depth = 37

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Log model manually with input example (optional if you want custom path/input)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example
    )

    # Log metrics manually
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
