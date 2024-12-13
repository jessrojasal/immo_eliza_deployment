import pandas as pd
import numpy as np
import os
import time
import pickle
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class Get_Regression_Metrics:
    """
    A class to compute and display regression evaluation metrics for
    both training and testing datasets.

    Metrics included:
    - Mean Absolute Error (MAE)
    - Root Mean Square Error (RMSE)
    - R² (Coefficient of Determination)
    - Mean Absolute Percentage Error (MAPE)
    - Symmetrical Mean Absolute Percentage Error (sMAPE)
    """

    def __init__(self, y_train, y_train_pred, y_test, y_test_pred):
        """
        Initializes Get_Regression_Metrics class with a actual and predicted values for test and train.

        :param y_train: Actual target values for the training set.
        :param y_train_pred: Predicted target values for the training set.
        :param y_test: Actual target values for the testing set.
        :param y_test_pred: Predicted target values for the testing set.
        """
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        self.y_test = y_test
        self.y_test_pred = y_test_pred

    def get_mae(self):
        """
        Mean absolute error (MAE) is an average of the absolute errors.
        MAE is always greater than or equal to 0 and MAE value 0 represents perfect fit.
        The MAE units are the same as the predicted target.
        """
        self.mae_train = mean_absolute_error(self.y_train, self.y_train_pred)
        self.mae_test = mean_absolute_error(self.y_test, self.y_test_pred)
        print(f"MAE(train): {self.mae_train:.3f}")
        print(f"MAE(test): {self.mae_test:.3f}")

    def get_rmse(self):
        """
        Root Mean Square Error (RMSE) represents the square root of the variance of the residuals.
        The smaller the RMSE, the closer your model's predictions are to reality.
        RMSE is expressed in the same unit as the predicted values
        """
        self.rmse_train = root_mean_squared_error(self.y_train, self.y_train_pred)
        self.rmse_test = root_mean_squared_error(self.y_test, self.y_test_pred)
        print(f"RMSE(train): {self.rmse_train:.3f}")
        print(f"RMSE(test): {self.rmse_test:.3f}")

    def get_r2(self):
        """
        R² (coefficient of determination) regression score function.
        R² is an element of [0, 1].
        Best possible score is 1.0 and it can be negative (the model can be arbitrarily worse).
        """
        self.r2_train = r2_score(self.y_train, self.y_train_pred)
        self.r2_test = r2_score(self.y_test, self.y_test_pred)
        print(f"R²(train): {self.r2_train:.3f}")
        print(f"R²(test): {self.r2_test:.3f}")

    def get_mape(self):
        """
        Mean absolute percentage error (MAPE) regression loss.
        It measures accuracy as a percentage.
        Lower values of MAPE indicate higher accuracy.
        """
        self.mape_train = mean_absolute_percentage_error(
            self.y_train, self.y_train_pred
        )
        self.mape_test = mean_absolute_percentage_error(self.y_test, self.y_test_pred)
        print(f"MAPE (train): {self.mape_train:.3f}")
        print(f"MAPE (test): {self.mape_test:.3f}")

    def get_smape(self):
        """
        Symmetrical mean absolute percentage error (sMAPE)
        is an accuracy measure based on percentage (or relative) errors.
        The resulting score ranges between 0 and 1, where a score of 0 indicates a perfect match.
        """

        def smape(actual, forecast):
            actual = np.array(actual)
            forecast = np.array(forecast)
            smape = (
                np.mean(
                    2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))
                )
                * 100
            )
            return smape

        self.smape_train = smape(self.y_train, self.y_train_pred)
        self.smape_test = smape(self.y_test, self.y_test_pred)
        print(f"sMAPE (train): {self.smape_train:.3f}")
        print(f"sMAPE (test): {self.smape_test:.3f}")

    def get_all_metrics(self):
        """
        Get all the metrics.
        """
        self.get_mae()
        self.get_rmse()
        self.get_r2()
        self.get_mape()
        self.get_smape()


class SplitData:
    """
    The SplitData class splits datasets into training and testing sets and handles encoding for categorical features.
    """

    def __init__(
        self,
        dataframe,
        target_column,
        categorical_columns=None,
        drop_columns=None,
        test_size=0.2,
        random_state=42,
    ):
        """
        Initialize the SplitData class.

        :param dataframe: The pandas DataFrame containing the dataset.
        :param target_column: The name of the target column to predict.
        :param categorical_columns: List of categorical columns to be label-encoded. Default is None.
        :param drop_columns: List of columns to drop from features. Default is None.
        :param test_size: Proportion of the dataset to include in the test split. Default is 0.2 (20%).
        :param random_state: Random state for reproducibility. Default is 42.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.categorical_columns = categorical_columns if categorical_columns else []
        self.drop_columns = drop_columns if drop_columns else []
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}

    def encode_categorical_features(self):
        """
        Encodes categorical features using LabelEncoder.
        """
        for column in self.categorical_columns:
            if column in self.dataframe.columns:
                le = LabelEncoder()
                self.dataframe[column] = le.fit_transform(self.dataframe[column])
                self.label_encoders[column] = le
                print(f"Encoded column: {column}")

    def split(self):
        """
        Splits the data into training and testing sets.

        :return: X_train: Training features.
        :return: X_test: Testing features.
        :return: y_train: Training target values.
        :return: y_test: Testing target values.
        """
        # Encode categorical features
        self.encode_categorical_features()

        # Drop specified columns and separate target
        X = self.dataframe.drop(
            columns=[self.target_column] + self.drop_columns, axis=1
        )
        y = self.dataframe[self.target_column].values.reshape(-1, 1)

        # Apply square root transformation to the target
        y_transformed = np.sqrt(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=self.test_size, random_state=self.random_state
        )
        print("Dataset split into training and testing sets with an 80/20 ratio")
        return X_train, X_test, y_train, y_test

    def save_encoders(self, directory="."):
        """
        Save the fitted LabelEncoder instances to disk for later use.

        :param directory: Directory where encoders will be saved (default is current folder).
        """
        for column, encoder in self.label_encoders.items():
            filepath = os.path.join(directory, f"{column}_encoder.pkl")
            with open(filepath, "wb") as file:
                pickle.dump(encoder, file)
            print(f"Encoder for '{column}' saved to {filepath}")


class ModelRandomForestRegressor:
    def __init__(
        self,
        random_state=None,
        n_estimators=None,
        min_samples_split=None,
        min_samples_leaf=None,
        max_leaf_nodes=None,
        max_depth=None,
    ):
        """
        Initialize the ModelRandomForestRegressor class.

        :param random_state: Random state for reproducibility.
        :param n_estimators: The number of trees in the forest.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        :param max_leaf_nodes: Maximum number of leaf nodes.
        :param max_depth: Maximum depth of the trees.
        """
        self.model = RandomForestRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
        )

    def train(self, X_train, y_train):
        """
        Train the Random Forest Regressor model.
        Measure training time for Random Forest

        :param X_train: Training features.
        :param y_train: Training target values.
        """
        start_time = time.time()
        self.model = self.model.fit(X_train, y_train.ravel())
        rf_training_time = time.time() - start_time
        print(f"Random Forest Training Time: {rf_training_time:.3f} seconds")
        return self.model

    def save_model(self, filename="random_forest_regressor_model.pkl"):
        """
        Save the trained model to a file.

        :param filename: File where the model will be saved.
        """
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {filename}")

    def predict(self, X):
        """
        Make predictions using the trained model.
        Measure inference time for Random Forest.

        :param X: Features to predict on.
        :return Predictions from the model.
        """
        start_time = time.time()
        self_pred = self.model.predict(X)
        rf_inference_time = time.time() - start_time
        print(f"Random Forest Inference Time: {rf_inference_time:.3f} seconds")
        return self_pred

    def evaluate(self, X_train, X_test, y_train, y_test):
        """
        Evaluate the models performance and calculate metrics.

        :param X_train: Training features.
        :param X_test: Testing features.
        :param y_train: Training target values.
        :param y_test: Testing target values.
        :return A dictionary of metrics for both training and testing sets.
        """
        # Flatten y_train and y_test to 1D arrays
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        # Predictions
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        # Transform predictions and actual values back to the original scale (squared)
        y_train_pred_original = y_train_pred**2
        y_test_pred_original = y_test_pred**2
        y_train_original = y_train**2
        y_test_original = y_test**2

        # Get metrics
        metrics = Get_Regression_Metrics(
            y_train_original,
            y_train_pred_original,
            y_test_original,
            y_test_pred_original,
        )
        print(f"Random Forest Regressor metrics: {metrics}")
        return metrics.get_all_metrics()
