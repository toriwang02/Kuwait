import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)


class ETFCreationPipeline:
    def __init__(self, index_csv, shares_csv):
        self.index_data = pd.read_csv(index_csv)
        self.shares_data = pd.read_csv(shares_csv)
        self.models = {}

    def preprocess_data(self):
        # Handle missing values
        self.index_data.fillna(method="ffill", inplace=True)
        self.shares_data.fillna(
            method="ffill", inplace=True
        )

        # Normalize or standardize the data
        scaler = MinMaxScaler()
        self.index_data_scaled = pd.DataFrame(
            scaler.fit_transform(self.index_data),
            columns=self.index_data.columns,
        )
        self.shares_data_scaled = pd.DataFrame(
            scaler.fit_transform(self.shares_data),
            columns=self.shares_data.columns,
        )

    def feature_engineering(self):
        # Example of adding moving averages as features
        self.index_data_scaled["moving_avg"] = (
            self.index_data_scaled["price"]
            .rolling(window=10)
            .mean()
        )
        self.shares_data_scaled["moving_avg"] = (
            self.shares_data_scaled["price"]
            .rolling(window=10)
            .mean()
        )

        # Example of adding order imbalance as a feature using LOB data
        self.shares_data_scaled["order_imbalance"] = (
            self.shares_data_scaled["buy_volume"]
            - self.shares_data_scaled["sell_volume"]
        ) / (
            self.shares_data_scaled["buy_volume"]
            + self.shares_data_scaled["sell_volume"]
        )

        # Drop rows with NaN values created by rolling function
        self.index_data_scaled.dropna(inplace=True)
        self.shares_data_scaled.dropna(inplace=True)

    def train_time_series_model(self):
        # ARIMA/GARCH example
        model = sm.tsa.ARIMA(
            self.index_data_scaled["price"], order=(5, 1, 0)
        )
        self.models["ARIMA"] = model.fit()

    def train_tree_based_model(self):
        # Train Random Forest or LightGBM as an example
        X = self.shares_data_scaled.drop("price", axis=1)
        y = self.shares_data_scaled["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        self.models["RandomForest"] = rf_model

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Random Forest - MSE: {mse}, R2: {r2}")

    def train_neural_network(self):
        # LSTM example in PyTorch
        class LSTMModel(nn.Module):
            def __init__(
                self,
                input_size,
                hidden_layer_size,
                output_size,
            ):
                super(LSTMModel, self).__init__()
                self.hidden_layer_size = hidden_layer_size
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_layer_size,
                    batch_first=True,
                )
                self.linear = nn.Linear(
                    hidden_layer_size, output_size
                )

            def forward(self, input_seq):
                lstm_out, _ = self.lstm(input_seq)
                predictions = self.linear(lstm_out[:, -1])
                return predictions

        X = self.shares_data_scaled.drop(
            "price", axis=1
        ).values
        y = self.shares_data_scaled["price"].values

        X = X.reshape(
            (X.shape[0], X.shape[1], 1)
        )  # Reshape for LSTM

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_torch = torch.tensor(
            X_train, dtype=torch.float32
        )
        y_train_torch = torch.tensor(
            y_train, dtype=torch.float32
        )
        X_test_torch = torch.tensor(
            X_test, dtype=torch.float32
        )
        y_test_torch = torch.tensor(
            y_test, dtype=torch.float32
        )

        model = LSTMModel(
            input_size=X_train.shape[1],
            hidden_layer_size=50,
            output_size=1,
        )
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):  # Number of epochs
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train_torch)
            loss = loss_function(
                y_pred.view(-1), y_train_torch
            )
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        self.models["LSTM"] = model

    def evaluate_model(self):
        for name, model in self.models.items():
            if name in ["ARIMA"]:
                print(f"{name} AIC: {model.aic}")
            elif name in ["RandomForest"]:
                X = self.shares_data_scaled.drop(
                    "price", axis=1
                )
                y = self.shares_data_scaled["price"]
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    scoring="neg_mean_squared_error",
                    cv=5,
                )
                print(
                    f"{name} Cross-Validation MSE: {np.mean(scores)}"
                )
            elif name in ["LSTM"]:
                model.eval()
                with torch.no_grad():
                    y_pred_torch = model(
                        torch.tensor(
                            X.reshape(
                                (X.shape[0], X.shape[1], 1)
                            ),
                            dtype=torch.float32,
                        )
                    )
                y_pred = y_pred_torch.view(-1).numpy()
                mse = mean_squared_error(
                    self.shares_data_scaled["price"], y_pred
                )
                print(f"{name} MSE: {mse}")

    def run_pipeline(self):
        self.preprocess_data()
        self.feature_engineering()
        self.train_time_series_model()
        self.train_tree_based_model()
        self.train_neural_network()
        self.evaluate_model()


# Example usage
pipeline = ETFCreationPipeline(
    "index_data.csv", "shares_data.csv"
)
pipeline.run_pipeline()
