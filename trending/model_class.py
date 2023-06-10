import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
import mlflow
import mlflow.xgboost

class DailyReadsModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state)
        self.features = None
        self.column_transformer = None

    def train(self, features, daily_reads, test_size=0.2):
        self.features = features
        log_daily_reads = np.log1p(daily_reads)  # Apply log transformation to the daily reads

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, log_daily_reads, test_size=test_size,
                                                            random_state=self.model.random_state)

        # Preprocess numerical and categorical features separately
        numerical_features = ["numerical_feature_1", "numerical_feature_2"]
        categorical_features = ["categorical_feature"]

        numerical_transformer = "passthrough"
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Apply preprocessing to training data
        X_train_preprocessed = self.column_transformer.fit_transform(X_train)

        # Define hyperparameters to tune
        params = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300]
        }

        # Start MLflow run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'n_estimators': self.model.n_estimators,
                'random_state': self.model.random_state,
                'test_size': test_size
            })

            # Perform hyperparameter tuning
            for max_depth in params['max_depth']:
                for learning_rate in params['learning_rate']:
                    for n_estimators in params['n_estimators']:
                        # Create XGBoost model
                        model = xgb.XGBRegressor(
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators
                        )

                        # Train the model
                        model.fit(X_train_preprocessed, y_train)

                        # Evaluate on test data
                        X_test_preprocessed = self.column_transformer.transform(X_test)
                        y_pred = model.predict(X_test_preprocessed)

                        # Calculate and log evaluation metrics
                        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                        mlflow.log_metric("rmse", rmse)

            # Train the final model with the best hyperparameters
            best_model = xgb.XGBRegressor(
                max_depth=best_max_depth,
                learning_rate=best_learning_rate,
                n_estimators=best_n_estimators
            )
            best_model.fit(X_train_preprocessed, y_train)

            # Log the best hyperparameters and the final model
            mlflow.log_params({
                'max_depth': best_max_depth,
                'learning_rate': best_learning_rate,
                'n_estimators': best_n_estimators
            })
            mlflow.xgboost.log_model(best_model, "model")

    def predict(self, future_features, confidence=0.95
