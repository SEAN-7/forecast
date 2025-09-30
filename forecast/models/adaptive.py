"""Adaptive forecasting model that selects algorithms dynamically."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class AdaptiveModel:
    """Adaptive forecasting model that selects algorithms based on data characteristics."""
    
    def __init__(self, algorithm: str = 'linear_regression', parameters: Dict[str, Any] = None):
        """
        Initialize the adaptive model.
        
        Args:
            algorithm: Name of the forecasting algorithm to use
            parameters: Parameters for the selected algorithm
        """
        self.algorithm = algorithm
        self.parameters = parameters or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
    def prepare_features(
        self,
        data: pd.DataFrame,
        target_column: str = 'value',
        lags: List[int] = None,
        window_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training/prediction.
        
        Args:
            data: Time series data
            target_column: Name of the target column
            lags: Lag periods to include as features
            window_size: Window size for rolling statistics
            
        Returns:
            Tuple of (features, targets)
        """
        if lags is None:
            lags = [1, 2, 3, 7]  # Default lag periods
        
        # Adjust parameters for small datasets
        if len(data) < 10:
            lags = [1]  # Only use lag 1 for very small datasets
            window_size = min(window_size, len(data) // 2)  # Reduce window size
        
        df = data.copy()
        
        # Ensure we have a numeric target
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in data")
        
        # Create lag features
        feature_names = []
        for lag in lags:
            if lag < len(df):  # Only create lag if we have enough data
                col_name = f'lag_{lag}'
                df[col_name] = df[target_column].shift(lag)
                feature_names.append(col_name)
        
        # Create rolling statistics
        rolling_stats = ['mean', 'std', 'min', 'max']
        for stat in rolling_stats:
            if window_size <= len(df):  # Only create rolling stats if we have enough data
                col_name = f'rolling_{stat}_{window_size}'
                df[col_name] = getattr(df[target_column].rolling(window=window_size), stat)()
                feature_names.append(col_name)
        
        # Create time-based features if timestamp column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            feature_names.extend(['day_of_week', 'month', 'quarter'])
        
        # Create trend feature (position in sequence)
        df['trend'] = np.arange(len(df))
        feature_names.append('trend')
        
        self.feature_names = feature_names
        
        # Drop rows with NaN values (caused by lags and rolling windows)
        df_clean = df.dropna()
        
        # If no data remains, create minimal features
        if len(df_clean) == 0:
            # Use original data with just trend feature
            df_clean = df[['timestamp', target_column, 'trend']].copy() if 'timestamp' in df.columns else df[[target_column, 'trend']].copy()
            feature_names = ['trend']
            df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after feature engineering")
        
        X = df_clean[feature_names].values
        y = df_clean[target_column].values
        
        return X, y
    
    def fit(self, data: pd.DataFrame, target_column: str = 'value') -> 'AdaptiveModel':
        """
        Fit the adaptive model to the data.
        
        Args:
            data: Time series data
            target_column: Name of the target column
            
        Returns:
            Self for method chaining
        """
        X, y = self.prepare_features(data, target_column)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit the selected algorithm
        if self.algorithm == 'linear_regression':
            self.model = LinearRegression(**self.parameters)
        elif self.algorithm == 'random_forest':
            # Ensure valid parameters for RandomForest
            rf_params = self.parameters.copy()
            if 'n_estimators' in rf_params:
                rf_params['n_estimators'] = max(1, int(rf_params['n_estimators']))
            if 'max_depth' in rf_params and rf_params['max_depth'] is not None:
                rf_params['max_depth'] = max(1, int(rf_params['max_depth']))
            
            self.model = RandomForestRegressor(**rf_params)
        elif self.algorithm == 'arima':
            # For ARIMA, we'll use a simple AR model as approximation
            self.model = self._create_ar_model(X, y)
        elif self.algorithm == 'exponential_smoothing':
            # For exponential smoothing, we'll use weighted linear regression
            self.model = self._create_exponential_smoothing_model(X, y)
        elif self.algorithm == 'prophet':
            # For Prophet, we'll use trend + seasonal components
            self.model = self._create_prophet_approximation(X, y)
        elif self.algorithm == 'neural_network':
            # For neural network, we'll use a simple polynomial regression
            self.model = self._create_neural_network_approximation(X, y)
        else:
            # Default to linear regression
            self.model = LinearRegression()
        
        # Fit the model
        if hasattr(self.model, 'fit'):
            self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 1,
        target_column: str = 'value'
    ) -> Dict[str, Any]:
        """
        Make predictions using the fitted model.
        
        Args:
            data: Time series data (should include recent history)
            horizon: Number of periods to forecast ahead
            target_column: Name of the target column
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        prediction_data = data.copy()
        
        for step in range(horizon):
            # Prepare features for the current step
            try:
                X, _ = self.prepare_features(prediction_data, target_column)
                
                if len(X) == 0:
                    # If we can't create features, use the last value
                    last_value = prediction_data[target_column].iloc[-1]
                    predictions.append(last_value)
                else:
                    # Ensure features match training features
                    if X.shape[1] != len(self.feature_names):
                        # Pad or truncate features to match training
                        if X.shape[1] < len(self.feature_names):
                            padding = np.zeros((X.shape[0], len(self.feature_names) - X.shape[1]))
                            X = np.hstack([X, padding])
                        else:
                            X = X[:, :len(self.feature_names)]
                    
                    # Scale features and predict
                    X_scaled = self.scaler.transform(X[-1:])  # Use only the most recent observation
                    
                    if hasattr(self.model, 'predict'):
                        pred = self.model.predict(X_scaled)[0]
                    else:
                        pred = self.model(X_scaled[0])
                    
                    predictions.append(pred)
            except Exception:
                # Fallback: use last value
                last_value = prediction_data[target_column].iloc[-1]
                predictions.append(last_value)
            
            # Add the prediction to the data for next step
            new_row = prediction_data.iloc[-1:].copy()
            new_row[target_column] = predictions[-1]
            
            # Update timestamp if available
            if 'timestamp' in new_row.columns:
                last_timestamp = pd.to_datetime(new_row['timestamp'].iloc[0])
                new_row['timestamp'] = last_timestamp + pd.Timedelta(days=1)  # Assume daily frequency
            
            prediction_data = pd.concat([prediction_data, new_row], ignore_index=True)
        
        # Calculate prediction intervals (simple approximation)
        std_error = self._estimate_prediction_error(data, target_column)
        confidence_interval = 1.96 * std_error  # 95% confidence interval
        
        lower_bounds = [p - confidence_interval for p in predictions]
        upper_bounds = [p + confidence_interval for p in predictions]
        
        return {
            'predictions': predictions,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'algorithm_used': self.algorithm,
            'prediction_error_std': std_error
        }
    
    def evaluate(self, data: pd.DataFrame, target_column: str = 'value') -> Dict[str, float]:
        """
        Evaluate the model performance on the given data.
        
        Args:
            data: Time series data for evaluation
            target_column: Name of the target column
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X, y_true = self.prepare_features(data, target_column)
        
        if len(X) == 0:
            return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf')}
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X_scaled)
        else:
            y_pred = np.array([self.model(x) for x in X_scaled])
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'sample_size': len(y_true)
        }
    
    def _create_ar_model(self, X: np.ndarray, y: np.ndarray):
        """Create an autoregressive model approximation for ARIMA."""
        # Simple AR(1) model using the first lag feature
        if X.shape[1] >= 1:
            # Use only the first lag feature for AR(1)
            model = LinearRegression()
            model.fit(X[:, :1], y)  # Use only first feature (first lag)
            return model
        else:
            # Fallback to mean
            return lambda x: np.mean(y)
    
    def _create_exponential_smoothing_model(self, X: np.ndarray, y: np.ndarray):
        """Create exponential smoothing approximation."""
        # Weighted linear regression with exponential weights
        alpha = self.parameters.get('alpha', 0.3)
        weights = np.array([alpha ** i for i in range(len(y))][::-1])
        
        # Weighted linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        # Apply weights by repeating samples
        sample_weights = weights / np.sum(weights) * len(weights)
        
        model.fit(X, y, sample_weight=sample_weights)
        return model
    
    def _create_prophet_approximation(self, X: np.ndarray, y: np.ndarray):
        """Create Prophet-like model approximation."""
        # Linear trend + seasonal components
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def _create_neural_network_approximation(self, X: np.ndarray, y: np.ndarray):
        """Create neural network approximation using polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        degree = min(2, X.shape[1])  # Limit polynomial degree
        
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        model.fit(X, y)
        return model
    
    def _estimate_prediction_error(self, data: pd.DataFrame, target_column: str) -> float:
        """Estimate prediction error for confidence intervals."""
        if len(data) < 10:
            return data[target_column].std() if len(data) > 1 else 1.0
        
        # Simple error estimation using recent data variance
        recent_data = data[target_column].tail(min(20, len(data)))
        return recent_data.std() if len(recent_data) > 1 else 1.0