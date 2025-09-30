"""Main forecaster class that orchestrates the AI-powered forecasting workflow."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import warnings
from datetime import datetime, timedelta

from .config import ForecastConfig
from ..data.processor import TimeSeriesProcessor
from ..agents.openai_agent import ForecastingAgent
from ..models.adaptive import AdaptiveModel


class Forecaster:
    """
    Main forecasting class that uses AI agents to adaptively select and configure
    forecasting algorithms based on time series characteristics and context.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize the forecaster.
        
        Args:
            config: Configuration object for forecasting
        """
        self.config = config or ForecastConfig()
        self.processor = TimeSeriesProcessor()
        self.agent = None
        self.model = None
        self.is_fitted = False
        self.last_validation_results = None
        self.forecast_metadata = {}
        
        # Initialize the AI agent if API key is available
        try:
            self.agent = ForecastingAgent(self.config)
        except ValueError as e:
            warnings.warn(f"AI agent not available: {e}. Will use fallback algorithm selection.")
            self.agent = None
    
    def load_data(
        self,
        data: Union[pd.DataFrame, Dict, List],
        time_column: Optional[str] = None,
        value_column: Optional[str] = None,
        date_format: Optional[str] = None
    ) -> 'Forecaster':
        """
        Load time series data for forecasting.
        
        Args:
            data: Input data (DataFrame, dict, or list)
            time_column: Name of the time/date column
            value_column: Name of the value column
            date_format: Format string for parsing dates
            
        Returns:
            Self for method chaining
        """
        self.processor.load_data(data, time_column, value_column, date_format)
        return self
    
    def prepare_data(
        self,
        handle_missing: str = 'interpolate',
        handle_duplicates: str = 'mean',
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0
    ) -> 'Forecaster':
        """
        Prepare and clean the time series data.
        
        Args:
            handle_missing: How to handle missing values ('interpolate', 'forward_fill', 'drop')
            handle_duplicates: How to handle duplicate timestamps ('mean', 'first', 'last')
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Self for method chaining
        """
        # Validate data first
        self.last_validation_results = self.processor.validate_data()
        
        if not self.last_validation_results['is_valid']:
            raise ValueError(f"Data validation failed: {self.last_validation_results['errors']}")
        
        # Log warnings
        for warning in self.last_validation_results['warnings']:
            warnings.warn(f"Data warning: {warning}")
        
        # Clean the data
        self.processor.clean_data(
            handle_missing=handle_missing,
            handle_duplicates=handle_duplicates,
            remove_outliers=remove_outliers,
            outlier_threshold=outlier_threshold
        )
        
        return self
    
    def fit(
        self,
        context: str = "",
        algorithm: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> 'Forecaster':
        """
        Fit the forecasting model to the data.
        
        Args:
            context: Contextual information about the forecasting scenario
            algorithm: Specific algorithm to use (if None, AI agent will select)
            parameters: Specific parameters to use (if None, AI agent will optimize)
            
        Returns:
            Self for method chaining
        """
        if self.processor.processed_data is None:
            raise ValueError("No processed data available. Call prepare_data() first.")
        
        # Extract features from the time series
        features = self.processor.get_features()
        
        # Store metadata
        self.forecast_metadata = {
            'features': features,
            'context': context,
            'fit_timestamp': datetime.now(),
            'data_stats': self.last_validation_results['stats'] if self.last_validation_results else {}
        }
        
        # Select algorithm and parameters using AI agent (if available) or fallback
        if algorithm is None:
            if self.agent:
                try:
                    selection_result = self.agent.select_algorithm(
                        features, context, self.config.max_forecast_horizon
                    )
                    algorithm = selection_result['algorithm']
                    self.forecast_metadata['algorithm_selection'] = selection_result
                except Exception as e:
                    warnings.warn(f"AI agent algorithm selection failed: {e}. Using fallback.")
                    algorithm = self._fallback_algorithm_selection(features)
            else:
                algorithm = self._fallback_algorithm_selection(features)
        
        if parameters is None:
            if self.agent:
                try:
                    param_result = self.agent.optimize_parameters(algorithm, features, context)
                    parameters = param_result['parameters']
                    self.forecast_metadata['parameter_optimization'] = param_result
                except Exception as e:
                    warnings.warn(f"AI agent parameter optimization failed: {e}. Using defaults.")
                    parameters = {}
            else:
                parameters = {}
        
        # Interpret context if AI agent is available
        if self.agent and context:
            try:
                context_result = self.agent.interpret_context(context)
                self.forecast_metadata['context_interpretation'] = context_result
            except Exception as e:
                warnings.warn(f"AI agent context interpretation failed: {e}")
        
        # Create and fit the model
        self.model = AdaptiveModel(algorithm, parameters)
        self.model.fit(self.processor.processed_data)
        
        self.is_fitted = True
        self.forecast_metadata['algorithm_used'] = algorithm
        self.forecast_metadata['parameters_used'] = parameters
        
        return self
    
    def predict(
        self,
        horizon: int = 1,
        confidence_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon: Number of periods to forecast ahead
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecasts and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        if horizon > self.config.max_forecast_horizon:
            warnings.warn(
                f"Requested horizon ({horizon}) exceeds maximum ({self.config.max_forecast_horizon}). "
                f"Using maximum horizon."
            )
            horizon = self.config.max_forecast_horizon
        
        confidence_level = confidence_level or self.config.confidence_level
        
        # Make predictions
        prediction_result = self.model.predict(
            self.processor.processed_data,
            horizon=horizon
        )
        
        # Create forecast dates
        last_date = self.processor.processed_data['timestamp'].iloc[-1]
        frequency = self.processor._infer_frequency()
        
        forecast_dates = []
        for i in range(1, horizon + 1):
            if frequency == 'daily':
                forecast_dates.append(last_date + timedelta(days=i))
            elif frequency == 'weekly':
                forecast_dates.append(last_date + timedelta(weeks=i))
            elif frequency == 'monthly':
                forecast_dates.append(last_date + pd.DateOffset(months=i))
            elif frequency == 'yearly':
                forecast_dates.append(last_date + pd.DateOffset(years=i))
            else:
                forecast_dates.append(last_date + timedelta(days=i))
        
        # Prepare result
        result = {
            'forecasts': prediction_result['predictions'],
            'lower_bounds': prediction_result['lower_bounds'],
            'upper_bounds': prediction_result['upper_bounds'],
            'dates': forecast_dates,
            'confidence_level': confidence_level,
            'algorithm_used': prediction_result['algorithm_used'],
            'prediction_error_std': prediction_result['prediction_error_std'],
            'metadata': self.forecast_metadata.copy()
        }
        
        return result
    
    def evaluate(self, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate the forecasting model performance.
        
        Args:
            test_data: Optional test data for evaluation. If None, uses cross-validation.
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation. Call fit() first.")
        
        if test_data is not None:
            # Evaluate on provided test data
            test_processor = TimeSeriesProcessor()
            test_processor.load_data(test_data)
            test_processor.clean_data()
            
            metrics = self.model.evaluate(test_processor.processed_data)
        else:
            # Use cross-validation on training data
            if self.config.enable_cross_validation:
                metrics = self._cross_validate()
            else:
                # Simple holdout validation
                metrics = self.model.evaluate(self.processor.processed_data)
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from the fitted model.
        
        Returns:
            Dictionary with feature importance information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance.")
        
        # For models that support feature importance
        if hasattr(self.model.model, 'feature_importances_'):
            importances = self.model.model.feature_importances_
            feature_names = self.model.feature_names
            
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return {
                'feature_importance': dict(sorted_importance),
                'top_features': [name for name, _ in sorted_importance[:5]]
            }
        elif hasattr(self.model.model, 'coef_'):
            # For linear models, use coefficients as importance
            coefficients = np.abs(self.model.model.coef_)
            feature_names = self.model.feature_names
            
            importance_dict = dict(zip(feature_names, coefficients))
            sorted_importance = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return {
                'feature_importance': dict(sorted_importance),
                'top_features': [name for name, _ in sorted_importance[:5]]
            }
        else:
            return {
                'feature_importance': {},
                'top_features': [],
                'note': 'Feature importance not available for this algorithm'
            }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded and processed data.
        
        Returns:
            Dictionary with data summary
        """
        if self.processor.data is None:
            return {'error': 'No data loaded'}
        
        summary = {
            'validation_results': self.last_validation_results,
            'features': self.processor.get_features() if self.processor.processed_data is not None else None,
            'forecast_metadata': self.forecast_metadata
        }
        
        return summary
    
    def _fallback_algorithm_selection(self, features: Dict[str, Any]) -> str:
        """Fallback algorithm selection when AI agent is not available."""
        length = features.get('length', 0)
        seasonality = features.get('seasonality', {})
        stationarity = features.get('stationarity', {})
        
        # Simple rule-based selection
        if length < 20:
            return 'linear_regression'
        elif seasonality.get('has_seasonality', False):
            return 'exponential_smoothing'
        elif not stationarity.get('is_stationary', True):
            return 'arima'
        elif length > 100:
            return 'random_forest'
        else:
            return 'linear_regression'
    
    def _cross_validate(self) -> Dict[str, float]:
        """Perform cross-validation on the training data."""
        data = self.processor.processed_data
        n_folds = min(self.config.cv_folds, len(data) // 4)  # Ensure reasonable fold size
        
        if n_folds < 2:
            # Fallback to simple evaluation
            return self.model.evaluate(data)
        
        fold_size = len(data) // n_folds
        metrics_list = []
        
        for i in range(n_folds):
            # Split data
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(data)
            
            # Use earlier data for training, later for testing
            train_data = data.iloc[:start_idx + fold_size//2]
            test_data = data.iloc[start_idx + fold_size//2:end_idx]
            
            if len(train_data) < 5 or len(test_data) < 1:
                continue
            
            # Create temporary model
            temp_model = AdaptiveModel(
                self.model.algorithm,
                self.model.parameters
            )
            
            try:
                temp_model.fit(train_data)
                fold_metrics = temp_model.evaluate(test_data)
                metrics_list.append(fold_metrics)
            except Exception:
                continue
        
        if not metrics_list:
            # Fallback to simple evaluation
            return self.model.evaluate(data)
        
        # Average metrics across folds
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = float('inf')
        
        avg_metrics['cv_folds'] = len(metrics_list)
        return avg_metrics