"""Basic tests for the forecast toolkit."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forecast.core.forecaster import Forecaster
from forecast.core.config import ForecastConfig
from forecast.data.processor import TimeSeriesProcessor
from forecast.models.adaptive import AdaptiveModel
from forecast.utils.helpers import generate_sample_data, validate_data_quality


class TestTimeSeriesProcessor:
    """Test the TimeSeriesProcessor class."""
    
    def test_load_data_from_dataframe(self):
        """Test loading data from DataFrame."""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        
        processor = TimeSeriesProcessor()
        processor.load_data(data, time_column='date')
        
        assert processor.data is not None
        assert 'timestamp' in processor.data.columns
        assert 'value' in processor.data.columns
        assert len(processor.data) == 10
    
    def test_load_data_from_list(self):
        """Test loading data from list."""
        values = [1, 2, 3, 4, 5]
        
        processor = TimeSeriesProcessor()
        processor.load_data(values)
        
        assert processor.data is not None
        assert len(processor.data) == 5
        assert processor.data['value'].tolist() == values
    
    def test_validate_data(self):
        """Test data validation."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        
        processor = TimeSeriesProcessor()
        processor.load_data(data)
        validation = processor.validate_data()
        
        assert validation['is_valid'] is True
        assert 'stats' in validation
        assert validation['stats']['length'] == 10
    
    def test_clean_data(self):
        """Test data cleaning."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10),
            'value': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10]
        })
        
        processor = TimeSeriesProcessor()
        processor.load_data(data)
        processor.clean_data(handle_missing='interpolate')
        
        assert processor.processed_data is not None
        assert not processor.processed_data['value'].isna().any()
    
    def test_get_features(self):
        """Test feature extraction."""
        data = generate_sample_data(length=50)
        
        processor = TimeSeriesProcessor()
        processor.load_data(data)
        processor.clean_data()
        features = processor.get_features()
        
        assert isinstance(features, dict)
        assert 'length' in features
        assert 'trend' in features
        assert 'seasonality' in features
        assert features['length'] == 50


class TestAdaptiveModel:
    """Test the AdaptiveModel class."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = AdaptiveModel('linear_regression')
        assert model.algorithm == 'linear_regression'
        assert not model.is_fitted
    
    def test_model_fitting(self):
        """Test model fitting."""
        data = generate_sample_data(length=30)
        
        model = AdaptiveModel('linear_regression')
        model.fit(data)
        
        assert model.is_fitted is True
        assert model.model is not None
    
    def test_model_prediction(self):
        """Test model prediction."""
        data = generate_sample_data(length=30)
        
        model = AdaptiveModel('linear_regression')
        model.fit(data)
        
        predictions = model.predict(data, horizon=5)
        
        assert 'predictions' in predictions
        assert len(predictions['predictions']) == 5
        assert 'lower_bounds' in predictions
        assert 'upper_bounds' in predictions
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        data = generate_sample_data(length=30)
        
        model = AdaptiveModel('linear_regression')
        model.fit(data)
        
        metrics = model.evaluate(data)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert isinstance(metrics['mae'], float)


class TestForecaster:
    """Test the main Forecaster class."""
    
    def test_forecaster_creation(self):
        """Test forecaster creation."""
        forecaster = Forecaster()
        assert forecaster.config is not None
        assert not forecaster.is_fitted
    
    def test_forecaster_workflow_without_openai(self):
        """Test complete forecaster workflow without OpenAI."""
        # Create sample data
        data = generate_sample_data(length=50)
        
        # Create forecaster without OpenAI key
        config = ForecastConfig(openai_api_key=None)
        forecaster = Forecaster(config)
        
        # Load and prepare data
        forecaster.load_data(data)
        forecaster.prepare_data()
        
        # Fit model (will use fallback algorithm selection)
        forecaster.fit(context="Test forecasting scenario")
        
        assert forecaster.is_fitted is True
        
        # Make predictions
        results = forecaster.predict(horizon=7)
        
        assert 'forecasts' in results
        assert len(results['forecasts']) == 7
        assert 'dates' in results
        assert 'lower_bounds' in results
        assert 'upper_bounds' in results
        
        # Evaluate model
        evaluation = forecaster.evaluate()
        assert 'mae' in evaluation
        
        # Get data summary
        summary = forecaster.get_data_summary()
        assert 'validation_results' in summary
    
    def test_forecaster_with_context(self):
        """Test forecaster with contextual information."""
        data = generate_sample_data(length=40)
        
        config = ForecastConfig(openai_api_key=None)
        forecaster = Forecaster(config)
        
        context = """
        This is sales data for a retail company. There was a major marketing campaign 
        in the last month which increased sales significantly. We expect this trend 
        to continue for the next few weeks.
        """
        
        forecaster.load_data(data)
        forecaster.prepare_data()
        forecaster.fit(context=context)
        
        results = forecaster.predict(horizon=5)
        
        assert results['metadata']['context'] == context
    
    def test_forecaster_algorithm_selection(self):
        """Test manual algorithm selection."""
        data = generate_sample_data(length=30)
        
        config = ForecastConfig(openai_api_key=None)
        forecaster = Forecaster(config)
        
        forecaster.load_data(data)
        forecaster.prepare_data()
        forecaster.fit(algorithm='random_forest')
        
        assert forecaster.forecast_metadata['algorithm_used'] == 'random_forest'


class TestUtilities:
    """Test utility functions."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        data = generate_sample_data(length=100, trend=0.1, seasonality=True)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'timestamp' in data.columns
        assert 'value' in data.columns
        assert data['timestamp'].dtype.name.startswith('datetime')
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Create data with some quality issues
        data = pd.DataFrame({
            'value': [1, 2, np.nan, 4, 1000, 6, 7, np.nan, 9, 10]  # Missing values and outlier
        })
        
        quality = validate_data_quality(data)
        
        assert 'quality_score' in quality
        assert 'missing_count' in quality
        assert 'outlier_count' in quality
        assert quality['missing_count'] == 2
        assert quality['outlier_count'] > 0
    
    def test_data_quality_excellent(self):
        """Test data quality with excellent data."""
        data = pd.DataFrame({
            'value': np.random.normal(100, 10, 100)  # Clean normal data
        })
        
        quality = validate_data_quality(data)
        
        assert quality['quality_score'] > 80  # Should be high quality
        assert quality['missing_count'] == 0


class TestConfig:
    """Test configuration handling."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ForecastConfig()
        
        assert config.max_forecast_horizon == 30
        assert config.confidence_level == 0.95
        assert config.min_data_points == 10
        assert 'linear_regression' in config.available_algorithms
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ForecastConfig(
            max_forecast_horizon=60,
            confidence_level=0.90,
            openai_model="gpt-3.5-turbo"
        )
        
        assert config.max_forecast_horizon == 60
        assert config.confidence_level == 0.90
        assert config.openai_model == "gpt-3.5-turbo"


# Pytest fixtures
@pytest.fixture
def sample_data():
    """Fixture providing sample time series data."""
    return generate_sample_data(length=50, trend=0.1, seasonality=True)


@pytest.fixture
def forecaster_without_openai():
    """Fixture providing a forecaster without OpenAI integration."""
    config = ForecastConfig(openai_api_key=None)
    return Forecaster(config)


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    print("Running basic tests...")
    
    # Test 1: TimeSeriesProcessor
    print("\n1. Testing TimeSeriesProcessor...")
    processor = TimeSeriesProcessor()
    test_data = generate_sample_data(30)
    processor.load_data(test_data)
    validation = processor.validate_data()
    print(f"   Data validation: {'PASS' if validation['is_valid'] else 'FAIL'}")
    
    processor.clean_data()
    features = processor.get_features()
    print(f"   Feature extraction: {'PASS' if features else 'FAIL'}")
    
    # Test 2: AdaptiveModel
    print("\n2. Testing AdaptiveModel...")
    model = AdaptiveModel('linear_regression')
    model.fit(processor.processed_data)
    print(f"   Model fitting: {'PASS' if model.is_fitted else 'FAIL'}")
    
    predictions = model.predict(processor.processed_data, horizon=3)
    print(f"   Model prediction: {'PASS' if predictions['predictions'] else 'FAIL'}")
    
    # Test 3: Forecaster
    print("\n3. Testing Forecaster...")
    config = ForecastConfig(openai_api_key=None)
    forecaster = Forecaster(config)
    
    forecaster.load_data(test_data)
    forecaster.prepare_data()
    forecaster.fit(context="Test scenario")
    print(f"   Forecaster fitting: {'PASS' if forecaster.is_fitted else 'FAIL'}")
    
    results = forecaster.predict(horizon=5)
    print(f"   Forecaster prediction: {'PASS' if len(results['forecasts']) == 5 else 'FAIL'}")
    
    # Test 4: Utilities
    print("\n4. Testing utilities...")
    quality = validate_data_quality(test_data)
    print(f"   Data quality check: {'PASS' if quality['quality_score'] > 0 else 'FAIL'}")
    
    print("\nAll basic tests completed!")