"""Time series data processing and validation."""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import warnings


class TimeSeriesProcessor:
    """Process and validate time series data for forecasting."""
    
    def __init__(self):
        """Initialize the processor."""
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
    
    def load_data(
        self,
        data: Union[pd.DataFrame, Dict, List],
        time_column: Optional[str] = None,
        value_column: Optional[str] = None,
        date_format: Optional[str] = None
    ) -> 'TimeSeriesProcessor':
        """
        Load time series data from various formats.
        
        Args:
            data: Input data (DataFrame, dict, or list)
            time_column: Name of the time/date column
            value_column: Name of the value column
            date_format: Format string for parsing dates
            
        Returns:
            Self for method chaining
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, dict):
            self.data = pd.DataFrame(data)
        elif isinstance(data, list):
            # Assume it's a list of values with default time index
            self.data = pd.DataFrame({
                'timestamp': pd.date_range(
                    start='2020-01-01',
                    periods=len(data),
                    freq='D'
                ),
                'value': data
            })
        else:
            raise ValueError("Unsupported data type. Use DataFrame, dict, or list.")
        
        # Standardize column names
        if time_column and time_column in self.data.columns:
            self.data = self.data.rename(columns={time_column: 'timestamp'})
        if value_column and value_column in self.data.columns:
            self.data = self.data.rename(columns={value_column: 'value'})
        
        # Ensure we have timestamp and value columns
        if 'timestamp' not in self.data.columns:
            if self.data.index.name in ['timestamp', 'date', 'time']:
                self.data = self.data.reset_index()
            else:
                # Create default timestamp column
                self.data['timestamp'] = pd.date_range(
                    start='2020-01-01',
                    periods=len(self.data),
                    freq='D'
                )
        
        if 'value' not in self.data.columns:
            # Try to find a numeric column
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.data = self.data.rename(columns={numeric_cols[0]: 'value'})
            else:
                raise ValueError("No numeric column found for 'value'")
        
        # Parse timestamps
        if date_format:
            self.data['timestamp'] = pd.to_datetime(
                self.data['timestamp'],
                format=date_format
            )
        else:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        return self
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the time series data and return validation results.
        
        Returns:
            Dictionary with validation results and warnings
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check minimum data points
        if len(self.data) < 3:
            validation_results['errors'].append(
                "Insufficient data: need at least 3 data points"
            )
            validation_results['is_valid'] = False
        
        # Check for missing values
        missing_values = self.data['value'].isna().sum()
        if missing_values > 0:
            validation_results['warnings'].append(
                f"Found {missing_values} missing values"
            )
        
        # Check for duplicate timestamps
        duplicate_timestamps = self.data['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            validation_results['warnings'].append(
                f"Found {duplicate_timestamps} duplicate timestamps"
            )
        
        # Check time series regularity
        time_diffs = self.data['timestamp'].diff().dropna()
        if len(time_diffs.unique()) > 1:
            validation_results['warnings'].append(
                "Irregular time intervals detected"
            )
        
        # Calculate basic statistics
        validation_results['stats'] = {
            'length': len(self.data),
            'start_date': self.data['timestamp'].min(),
            'end_date': self.data['timestamp'].max(),
            'frequency': self._infer_frequency(),
            'missing_values': missing_values,
            'duplicate_timestamps': duplicate_timestamps,
            'mean_value': self.data['value'].mean(),
            'std_value': self.data['value'].std(),
            'min_value': self.data['value'].min(),
            'max_value': self.data['value'].max()
        }
        
        return validation_results
    
    def clean_data(
        self,
        handle_missing: str = 'interpolate',
        handle_duplicates: str = 'mean',
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0
    ) -> 'TimeSeriesProcessor':
        """
        Clean the time series data.
        
        Args:
            handle_missing: How to handle missing values ('interpolate', 'forward_fill', 'drop')
            handle_duplicates: How to handle duplicate timestamps ('mean', 'first', 'last')
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.processed_data = self.data.copy()
        
        # Handle duplicate timestamps
        if handle_duplicates == 'mean':
            self.processed_data = (
                self.processed_data
                .groupby('timestamp')
                .agg({'value': 'mean'})
                .reset_index()
            )
        elif handle_duplicates == 'first':
            self.processed_data = self.processed_data.drop_duplicates(
                subset=['timestamp'],
                keep='first'
            )
        elif handle_duplicates == 'last':
            self.processed_data = self.processed_data.drop_duplicates(
                subset=['timestamp'],
                keep='last'
            )
        
        # Handle missing values
        if handle_missing == 'interpolate':
            self.processed_data['value'] = (
                self.processed_data['value'].interpolate(method='linear')
            )
        elif handle_missing == 'forward_fill':
            self.processed_data['value'] = self.processed_data['value'].ffill()
        elif handle_missing == 'drop':
            self.processed_data = self.processed_data.dropna(subset=['value'])
        
        # Remove outliers if requested
        if remove_outliers:
            z_scores = np.abs(
                (self.processed_data['value'] - self.processed_data['value'].mean()) /
                self.processed_data['value'].std()
            )
            self.processed_data = self.processed_data[z_scores < outlier_threshold]
        
        # Sort by timestamp
        self.processed_data = self.processed_data.sort_values('timestamp').reset_index(drop=True)
        
        return self
    
    def get_features(self) -> Dict[str, Any]:
        """
        Extract features from the time series for the AI agent.
        
        Returns:
            Dictionary of extracted features
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data() first.")
        
        data = self.processed_data['value'].values
        
        features = {
            'length': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'trend': self._calculate_trend(),
            'seasonality': self._detect_seasonality(),
            'stationarity': self._test_stationarity(),
            'autocorrelation': self._calculate_autocorrelation(),
            'frequency': self._infer_frequency(),
            'missing_ratio': self.data['value'].isna().sum() / len(self.data),
        }
        
        return features
    
    def _infer_frequency(self) -> str:
        """Infer the frequency of the time series."""
        if self.data is None or len(self.data) < 2:
            return 'unknown'
        
        time_diffs = self.data['timestamp'].diff().dropna()
        most_common_diff = time_diffs.mode()
        
        if len(most_common_diff) == 0:
            return 'irregular'
        
        diff = most_common_diff.iloc[0]
        
        if diff == timedelta(days=1):
            return 'daily'
        elif diff == timedelta(days=7):
            return 'weekly'
        elif diff.days >= 28 and diff.days <= 31:
            return 'monthly'
        elif diff.days >= 365 and diff.days <= 366:
            return 'yearly'
        else:
            return f'{diff.total_seconds()}s'
    
    def _calculate_trend(self) -> float:
        """Calculate the trend in the time series."""
        if self.processed_data is None or len(self.processed_data) < 2:
            return 0.0
        
        x = np.arange(len(self.processed_data))
        y = self.processed_data['value'].values
        
        # Linear regression to find trend
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    
    def _detect_seasonality(self) -> Dict[str, Any]:
        """Detect seasonality in the time series."""
        if self.processed_data is None or len(self.processed_data) < 10:
            return {'has_seasonality': False, 'period': None}
        
        # Simple seasonality detection using autocorrelation
        values = self.processed_data['value'].values
        n = len(values)
        
        # Check for common seasonal periods
        periods_to_check = [7, 12, 24, 30, 365]  # Weekly, monthly, daily, yearly
        periods_to_check = [p for p in periods_to_check if p < n // 2]
        
        max_autocorr = 0
        best_period = None
        
        for period in periods_to_check:
            if period < n:
                autocorr = np.corrcoef(values[:-period], values[period:])[0, 1]
                if not np.isnan(autocorr) and autocorr > max_autocorr:
                    max_autocorr = autocorr
                    best_period = period
        
        has_seasonality = max_autocorr > 0.3  # Threshold for significant seasonality
        
        return {
            'has_seasonality': has_seasonality,
            'period': best_period,
            'strength': max_autocorr
        }
    
    def _test_stationarity(self) -> Dict[str, Any]:
        """Test for stationarity in the time series."""
        if self.processed_data is None or len(self.processed_data) < 10:
            return {'is_stationary': False, 'confidence': 0.0}
        
        # Simple stationarity test: compare variance of first and second half
        values = self.processed_data['value'].values
        n = len(values)
        
        first_half_var = np.var(values[:n//2])
        second_half_var = np.var(values[n//2:])
        
        # If variances are similar, likely stationary
        if first_half_var == 0 or second_half_var == 0:
            return {'is_stationary': False, 'confidence': 0.0}
        
        var_ratio = min(first_half_var, second_half_var) / max(first_half_var, second_half_var)
        
        is_stationary = var_ratio > 0.5  # Threshold for similar variances
        confidence = var_ratio
        
        return {
            'is_stationary': is_stationary,
            'confidence': confidence
        }
    
    def _calculate_autocorrelation(self, max_lags: int = 10) -> List[float]:
        """Calculate autocorrelation for different lags."""
        if self.processed_data is None or len(self.processed_data) < 10:
            return []
        
        values = self.processed_data['value'].values
        n = len(values)
        max_lags = min(max_lags, n // 4)  # Don't use more than 1/4 of the data
        
        autocorrelations = []
        for lag in range(1, max_lags + 1):
            if lag < n:
                autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrelations.append(autocorr)
                else:
                    autocorrelations.append(0.0)
        
        return autocorrelations
    
    def get_data(self, processed: bool = True) -> pd.DataFrame:
        """
        Get the time series data.
        
        Args:
            processed: Whether to return processed or raw data
            
        Returns:
            DataFrame with time series data
        """
        if processed and self.processed_data is not None:
            return self.processed_data.copy()
        elif self.data is not None:
            return self.data.copy()
        else:
            raise ValueError("No data available")