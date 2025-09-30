"""Utility functions for the forecast toolkit."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


def plot_forecast(
    result: Dict[str, Any],
    historical_data: Optional[pd.DataFrame] = None,
    title: str = "Forecast Results",
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot forecast results with confidence intervals.
    
    Args:
        result: Forecast result dictionary from Forecaster.predict()
        historical_data: Optional historical data to plot
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot historical data if provided
    if historical_data is not None:
        if 'timestamp' in historical_data.columns and 'value' in historical_data.columns:
            plt.plot(
                historical_data['timestamp'],
                historical_data['value'],
                label='Historical Data',
                color='blue',
                alpha=0.7
            )
    
    # Plot forecasts
    dates = result['dates']
    forecasts = result['forecasts']
    lower_bounds = result['lower_bounds']
    upper_bounds = result['upper_bounds']
    
    plt.plot(dates, forecasts, label='Forecast', color='red', linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(
        dates,
        lower_bounds,
        upper_bounds,
        alpha=0.3,
        color='red',
        label=f'{result["confidence_level"]*100:.0f}% Confidence Interval'
    )
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    feature_importance: Dict[str, Any],
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_importance: Feature importance dictionary
        top_n: Number of top features to plot
        figsize: Figure size
    """
    if not feature_importance.get('feature_importance'):
        print("No feature importance available")
        return
    
    importance_dict = feature_importance['feature_importance']
    
    # Get top N features
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    features, importances = zip(*sorted_features)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def generate_sample_data(
    length: int = 100,
    trend: float = 0.1,
    seasonality: bool = True,
    noise_level: float = 0.1,
    start_date: str = "2020-01-01"
) -> pd.DataFrame:
    """
    Generate sample time series data for testing.
    
    Args:
        length: Number of data points
        trend: Linear trend component
        seasonality: Whether to include seasonal component
        noise_level: Standard deviation of random noise
        start_date: Start date for the time series
        
    Returns:
        DataFrame with sample time series data
    """
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    
    # Linear trend
    trend_component = trend * np.arange(length)
    
    # Seasonal component
    if seasonality:
        seasonal_component = 5 * np.sin(2 * np.pi * np.arange(length) / 365.25)  # Yearly seasonality
        seasonal_component += 2 * np.sin(2 * np.pi * np.arange(length) / 7)     # Weekly seasonality
    else:
        seasonal_component = np.zeros(length)
    
    # Random noise
    noise = np.random.normal(0, noise_level, length)
    
    # Base level
    base_level = 10
    
    # Combine components
    values = base_level + trend_component + seasonal_component + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'value': values
    })


def evaluate_multiple_algorithms(
    data: pd.DataFrame,
    algorithms: List[str],
    context: str = "",
    horizon: int = 5,
    train_ratio: float = 0.8
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple forecasting algorithms on the same data.
    
    Args:
        data: Time series data
        algorithms: List of algorithms to test
        context: Contextual information
        horizon: Forecast horizon
        train_ratio: Ratio of data to use for training
        
    Returns:
        Dictionary with results for each algorithm
    """
    from ..core.forecaster import Forecaster
    
    # Split data
    split_point = int(len(data) * train_ratio)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    results = {}
    
    for algorithm in algorithms:
        try:
            # Create and fit forecaster
            forecaster = Forecaster()
            forecaster.load_data(train_data)
            forecaster.prepare_data()
            forecaster.fit(context=context, algorithm=algorithm)
            
            # Make predictions
            predictions = forecaster.predict(horizon=min(horizon, len(test_data)))
            
            # Evaluate if we have test data
            if len(test_data) > 0:
                evaluation = forecaster.evaluate(test_data)
            else:
                evaluation = {}
            
            results[algorithm] = {
                'predictions': predictions,
                'evaluation': evaluation,
                'feature_importance': forecaster.get_feature_importance()
            }
            
        except Exception as e:
            results[algorithm] = {
                'error': str(e),
                'predictions': None,
                'evaluation': {},
                'feature_importance': {}
            }
    
    return results


def create_forecast_report(
    forecast_result: Dict[str, Any],
    data_summary: Dict[str, Any],
    evaluation: Dict[str, Any] = None
) -> str:
    """
    Create a text report of forecast results.
    
    Args:
        forecast_result: Result from Forecaster.predict()
        data_summary: Result from Forecaster.get_data_summary()
        evaluation: Optional evaluation metrics
        
    Returns:
        Formatted text report
    """
    report = []
    report.append("FORECAST REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Data summary
    if 'validation_results' in data_summary:
        validation = data_summary['validation_results']
        if 'stats' in validation:
            stats = validation['stats']
            report.append("DATA SUMMARY:")
            report.append(f"  Data points: {stats.get('length', 'N/A')}")
            report.append(f"  Date range: {stats.get('start_date', 'N/A')} to {stats.get('end_date', 'N/A')}")
            report.append(f"  Frequency: {stats.get('frequency', 'N/A')}")
            report.append(f"  Missing values: {stats.get('missing_values', 'N/A')}")
            report.append("")
    
    # Algorithm selection
    metadata = forecast_result.get('metadata', {})
    algorithm = forecast_result.get('algorithm_used', 'Unknown')
    report.append(f"ALGORITHM SELECTED: {algorithm}")
    
    if 'algorithm_selection' in metadata:
        selection = metadata['algorithm_selection']
        report.append(f"  Confidence: {selection.get('confidence', 'N/A'):.2f}")
        report.append(f"  Reasoning: {selection.get('reasoning', 'N/A')}")
    
    report.append("")
    
    # Context interpretation
    if 'context_interpretation' in metadata:
        context = metadata['context_interpretation']
        report.append("CONTEXT ANALYSIS:")
        report.append(f"  Impact: {context.get('impact', 'N/A')}")
        report.append(f"  Key factors: {', '.join(context.get('factors', []))}")
        report.append("")
    
    # Forecast results
    report.append("FORECAST RESULTS:")
    forecasts = forecast_result.get('forecasts', [])
    dates = forecast_result.get('dates', [])
    
    for i, (date, forecast) in enumerate(zip(dates, forecasts)):
        report.append(f"  {date.strftime('%Y-%m-%d')}: {forecast:.2f}")
        if i >= 9:  # Limit to first 10 forecasts
            remaining = len(forecasts) - 10
            if remaining > 0:
                report.append(f"  ... and {remaining} more periods")
            break
    
    report.append("")
    
    # Evaluation metrics
    if evaluation:
        report.append("MODEL PERFORMANCE:")
        for metric, value in evaluation.items():
            if isinstance(value, float):
                report.append(f"  {metric.upper()}: {value:.4f}")
            else:
                report.append(f"  {metric.upper()}: {value}")
        report.append("")
    
    # Technical details
    report.append("TECHNICAL DETAILS:")
    report.append(f"  Confidence level: {forecast_result.get('confidence_level', 'N/A')}")
    report.append(f"  Prediction error std: {forecast_result.get('prediction_error_std', 'N/A'):.4f}")
    
    return "\n".join(report)


def export_forecast_csv(
    forecast_result: Dict[str, Any],
    filename: str = "forecast_results.csv"
) -> None:
    """
    Export forecast results to CSV file.
    
    Args:
        forecast_result: Result from Forecaster.predict()
        filename: Output filename
    """
    df = pd.DataFrame({
        'date': forecast_result['dates'],
        'forecast': forecast_result['forecasts'],
        'lower_bound': forecast_result['lower_bounds'],
        'upper_bound': forecast_result['upper_bounds']
    })
    
    df.to_csv(filename, index=False)
    print(f"Forecast results exported to {filename}")


def validate_data_quality(data: pd.DataFrame, column: str = 'value') -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        data: Input data
        column: Column to analyze
        
    Returns:
        Dictionary with quality assessment
    """
    if column not in data.columns:
        return {'error': f"Column '{column}' not found in data"}
    
    values = data[column]
    
    quality_report = {
        'total_points': len(values),
        'missing_count': values.isna().sum(),
        'missing_percentage': (values.isna().sum() / len(values)) * 100,
        'duplicate_count': values.duplicated().sum(),
        'zero_count': (values == 0).sum(),
        'negative_count': (values < 0).sum(),
        'infinite_count': np.isinf(values).sum(),
        'unique_values': values.nunique(),
        'mean': values.mean(),
        'median': values.median(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'outlier_count': 0,
        'quality_score': 0.0
    }
    
    # Detect outliers using IQR method
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    
    outlier_threshold_low = Q1 - 1.5 * IQR
    outlier_threshold_high = Q3 + 1.5 * IQR
    
    outliers = values[(values < outlier_threshold_low) | (values > outlier_threshold_high)]
    quality_report['outlier_count'] = len(outliers)
    quality_report['outlier_percentage'] = (len(outliers) / len(values)) * 100
    
    # Calculate quality score (0-100)
    score = 100
    score -= quality_report['missing_percentage'] * 2  # Penalize missing values
    score -= min(quality_report['outlier_percentage'] * 0.5, 20)  # Penalize outliers
    score -= min((quality_report['infinite_count'] / len(values)) * 100, 10)  # Penalize infinites
    
    quality_report['quality_score'] = max(0, score)
    
    # Quality assessment
    if quality_report['quality_score'] >= 90:
        quality_report['assessment'] = 'Excellent'
    elif quality_report['quality_score'] >= 75:
        quality_report['assessment'] = 'Good'
    elif quality_report['quality_score'] >= 60:
        quality_report['assessment'] = 'Fair'
    else:
        quality_report['assessment'] = 'Poor'
    
    return quality_report