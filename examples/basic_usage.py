"""
Example usage of the Forecast toolkit.

This script demonstrates how to use the forecasting toolkit for
time series prediction with AI-powered algorithm selection.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forecast import Forecaster, ForecastConfig
from forecast.utils.helpers import (
    generate_sample_data,
    plot_forecast,
    create_forecast_report,
    export_forecast_csv,
    validate_data_quality
)


def example_basic_forecasting():
    """Basic forecasting example without OpenAI integration."""
    print("=== Basic Forecasting Example ===\n")
    
    # Generate sample data
    print("1. Generating sample time series data...")
    data = generate_sample_data(
        length=100,
        trend=0.05,
        seasonality=True,
        noise_level=2.0,
        start_date="2023-01-01"
    )
    print(f"   Generated {len(data)} data points")
    print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Check data quality
    print("\n2. Checking data quality...")
    quality = validate_data_quality(data)
    print(f"   Quality score: {quality['quality_score']:.1f}/100 ({quality['assessment']})")
    print(f"   Missing values: {quality['missing_count']}")
    print(f"   Outliers: {quality['outlier_count']}")
    
    # Create forecaster (without OpenAI for this example)
    print("\n3. Creating forecaster...")
    config = ForecastConfig(
        openai_api_key=None,  # No OpenAI integration
        max_forecast_horizon=30,
        confidence_level=0.95
    )
    forecaster = Forecaster(config)
    
    # Load and prepare data
    print("\n4. Loading and preparing data...")
    forecaster.load_data(data)
    forecaster.prepare_data(
        handle_missing='interpolate',
        remove_outliers=False
    )
    
    # Get data summary
    summary = forecaster.get_data_summary()
    validation_stats = summary['validation_results']['stats']
    print(f"   Data length: {validation_stats['length']}")
    print(f"   Frequency: {validation_stats['frequency']}")
    print(f"   Mean value: {validation_stats['mean_value']:.2f}")
    
    # Fit the model with context
    print("\n5. Training forecasting model...")
    context = """
    This is simulated sales data for an e-commerce company. The data shows 
    a positive growth trend with seasonal patterns. There are weekly cycles 
    (higher sales on weekends) and annual cycles (holiday seasons).
    """
    
    forecaster.fit(context=context)
    print(f"   Algorithm selected: {forecaster.forecast_metadata['algorithm_used']}")
    
    # Make predictions
    print("\n6. Generating forecasts...")
    forecast_horizon = 14  # 2 weeks ahead
    results = forecaster.predict(horizon=forecast_horizon)
    
    print(f"   Generated {len(results['forecasts'])} forecasts")
    print(f"   Confidence level: {results['confidence_level']*100}%")
    
    # Display first few predictions
    print("\n   First 5 predictions:")
    for i in range(min(5, len(results['forecasts']))):
        date = results['dates'][i]
        forecast = results['forecasts'][i]
        lower = results['lower_bounds'][i]
        upper = results['upper_bounds'][i]
        print(f"     {date.strftime('%Y-%m-%d')}: {forecast:.2f} [{lower:.2f}, {upper:.2f}]")
    
    # Evaluate model
    print("\n7. Evaluating model performance...")
    evaluation = forecaster.evaluate()
    print(f"   MAE (Mean Absolute Error): {evaluation['mae']:.4f}")
    print(f"   RMSE (Root Mean Square Error): {evaluation['rmse']:.4f}")
    
    # Get feature importance
    feature_importance = forecaster.get_feature_importance()
    if feature_importance['top_features']:
        print(f"\n   Top features: {', '.join(feature_importance['top_features'][:3])}")
    
    # Create and display report
    print("\n8. Generating forecast report...")
    report = create_forecast_report(results, summary, evaluation)
    print("\n" + "="*50)
    print(report)
    print("="*50)
    
    # Export results
    print("\n9. Exporting results...")
    export_forecast_csv(results, "/tmp/basic_forecast_results.csv")
    
    return forecaster, results


def example_multiple_algorithms():
    """Example comparing multiple algorithms."""
    print("\n\n=== Multiple Algorithms Comparison ===\n")
    
    from forecast.utils.helpers import evaluate_multiple_algorithms
    
    # Generate data with clear seasonality
    print("1. Generating seasonal data...")
    data = generate_sample_data(
        length=200,
        trend=0.02,
        seasonality=True,
        noise_level=1.0
    )
    
    # Test multiple algorithms
    print("\n2. Testing multiple algorithms...")
    algorithms = ['linear_regression', 'random_forest', 'exponential_smoothing', 'arima']
    
    context = """
    Retail sales data with strong seasonal patterns and moderate growth trend.
    """
    
    results = evaluate_multiple_algorithms(
        data=data,
        algorithms=algorithms,
        context=context,
        horizon=7,
        train_ratio=0.8
    )
    
    # Display results
    print("\n3. Algorithm comparison results:")
    print(f"{'Algorithm':<20} {'MAE':<10} {'RMSE':<10} {'Status'}")
    print("-" * 50)
    
    for algorithm, result in results.items():
        if 'error' in result:
            print(f"{algorithm:<20} {'ERROR':<10} {'ERROR':<10} {result['error'][:20]}")
        else:
            mae = result['evaluation'].get('mae', 'N/A')
            rmse = result['evaluation'].get('rmse', 'N/A')
            mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else str(rmse)
            print(f"{algorithm:<20} {mae_str:<10} {rmse_str:<10} {'SUCCESS'}")
    
    return results


def example_with_custom_data():
    """Example using custom CSV data format."""
    print("\n\n=== Custom Data Example ===\n")
    
    # Create a custom dataset
    print("1. Creating custom dataset...")
    dates = pd.date_range('2022-01-01', periods=150, freq='D')
    
    # Simulate website traffic data
    base_traffic = 1000
    trend = np.linspace(0, 300, len(dates))  # Growing trend
    
    # Weekly seasonality (lower on weekends)
    weekly_pattern = np.where(
        pd.to_datetime(dates).dayofweek >= 5,  # Weekend
        -200,  # Lower traffic
        100    # Higher traffic
    )
    
    # Add some random events (marketing campaigns, holidays)
    events = np.zeros(len(dates))
    event_dates = [30, 60, 90, 120]  # Days with special events
    for event_day in event_dates:
        if event_day < len(dates):
            events[event_day:event_day+3] = 500  # 3-day boost
    
    # Combine all components
    noise = np.random.normal(0, 50, len(dates))
    traffic = base_traffic + trend + weekly_pattern + events + noise
    traffic = np.maximum(traffic, 0)  # Ensure non-negative
    
    custom_data = pd.DataFrame({
        'date': dates,
        'daily_visitors': traffic
    })
    
    print(f"   Created dataset with {len(custom_data)} days of traffic data")
    print(f"   Average daily visitors: {custom_data['daily_visitors'].mean():.0f}")
    
    # Forecast with custom data
    print("\n2. Forecasting website traffic...")
    forecaster = Forecaster()
    
    forecaster.load_data(
        custom_data,
        time_column='date',
        value_column='daily_visitors'
    )
    
    forecaster.prepare_data()
    
    context = """
    This is website traffic data showing daily visitor counts. The data has:
    - A steady growth trend due to SEO improvements
    - Weekly seasonality with lower traffic on weekends
    - Occasional traffic spikes from marketing campaigns
    
    We want to forecast traffic for the next 2 weeks to plan server capacity.
    """
    
    forecaster.fit(context=context)
    
    # Generate 2-week forecast
    results = forecaster.predict(horizon=14)
    
    print(f"   Algorithm used: {results['algorithm_used']}")
    print(f"   Next week average predicted traffic: {np.mean(results['forecasts'][:7]):.0f}")
    print(f"   Week after average predicted traffic: {np.mean(results['forecasts'][7:]):.0f}")
    
    # Show weekend vs weekday predictions
    print("\n3. Weekend vs Weekday prediction analysis:")
    for i, (date, forecast) in enumerate(zip(results['dates'], results['forecasts'])):
        day_type = "Weekend" if date.dayofweek >= 5 else "Weekday"
        print(f"   {date.strftime('%Y-%m-%d')} ({day_type}): {forecast:.0f} visitors")
    
    return forecaster, results, custom_data


def main():
    """Run all examples."""
    print("FORECAST TOOLKIT - EXAMPLE USAGE")
    print("=" * 60)
    
    try:
        # Run basic example
        forecaster1, results1 = example_basic_forecasting()
        
        # Run algorithm comparison
        comparison_results = example_multiple_algorithms()
        
        # Run custom data example
        forecaster2, results2, custom_data = example_with_custom_data()
        
        print("\n\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey takeaways:")
        print("- The toolkit can work without OpenAI API key using fallback algorithms")
        print("- Multiple data formats are supported (DataFrame, dict, list)")
        print("- Context information helps improve forecasting accuracy")
        print("- Built-in data quality checks and validation")
        print("- Easy algorithm comparison and evaluation")
        print("- Comprehensive reporting and export capabilities")
        
        print(f"\nResults exported to: /tmp/basic_forecast_results.csv")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)