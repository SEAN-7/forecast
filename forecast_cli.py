#!/usr/bin/env python3
"""
Simple CLI interface for the Forecast toolkit.

Usage:
    python forecast_cli.py --data data.csv --forecast-days 7
    python forecast_cli.py --example
"""

import argparse
import sys
import os
import pandas as pd

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from forecast import Forecaster, ForecastConfig
from forecast.utils.helpers import (
    generate_sample_data,
    create_forecast_report,
    export_forecast_csv,
    validate_data_quality
)


def main():
    parser = argparse.ArgumentParser(description='Forecast Toolkit CLI')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--time-column', type=str, default='timestamp', help='Name of time column')
    parser.add_argument('--value-column', type=str, default='value', help='Name of value column')
    parser.add_argument('--forecast-days', type=int, default=7, help='Number of days to forecast')
    parser.add_argument('--context', type=str, default='', help='Business context for forecasting')
    parser.add_argument('--algorithm', type=str, help='Specific algorithm to use')
    parser.add_argument('--output', type=str, default='forecast_results.csv', help='Output CSV file')
    parser.add_argument('--example', action='store_true', help='Run with example data')
    parser.add_argument('--no-openai', action='store_true', help='Disable OpenAI integration')
    
    args = parser.parse_args()
    
    # Configure OpenAI
    config = ForecastConfig()
    if args.no_openai:
        config.openai_api_key = None
    
    # Load data
    if args.example:
        print("Generating example data...")
        data = generate_sample_data(length=100, trend=0.1, seasonality=True)
        print(f"Generated {len(data)} data points")
    elif args.data:
        print(f"Loading data from {args.data}...")
        try:
            data = pd.read_csv(args.data)
            print(f"Loaded {len(data)} rows")
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1
    else:
        print("Error: Please provide --data or --example")
        return 1
    
    # Check data quality
    if args.value_column in data.columns:
        quality = validate_data_quality(data, args.value_column)
        print(f"Data quality: {quality['quality_score']:.1f}/100 ({quality['assessment']})")
    
    # Create forecaster
    print("Creating forecaster...")
    forecaster = Forecaster(config)
    
    # Load and prepare data
    try:
        forecaster.load_data(
            data,
            time_column=args.time_column if args.time_column in data.columns else None,
            value_column=args.value_column if args.value_column in data.columns else None
        )
        forecaster.prepare_data()
        print("Data loaded and prepared successfully")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return 1
    
    # Fit model
    print("Training forecasting model...")
    try:
        forecaster.fit(
            context=args.context,
            algorithm=args.algorithm
        )
        algorithm_used = forecaster.forecast_metadata['algorithm_used']
        print(f"Model trained using: {algorithm_used}")
    except Exception as e:
        print(f"Error training model: {e}")
        return 1
    
    # Generate forecasts
    print(f"Generating {args.forecast_days}-day forecast...")
    try:
        results = forecaster.predict(horizon=args.forecast_days)
        print(f"Generated {len(results['forecasts'])} predictions")
        
        # Show first few predictions
        print("\nForecast results:")
        for i, (date, forecast) in enumerate(zip(results['dates'], results['forecasts'])):
            lower = results['lower_bounds'][i]
            upper = results['upper_bounds'][i]
            print(f"  {date.strftime('%Y-%m-%d')}: {forecast:.2f} [{lower:.2f}, {upper:.2f}]")
        
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return 1
    
    # Evaluate model
    try:
        evaluation = forecaster.evaluate()
        print(f"\nModel performance:")
        print(f"  MAE: {evaluation['mae']:.4f}")
        print(f"  RMSE: {evaluation['rmse']:.4f}")
    except Exception as e:
        print(f"Warning: Could not evaluate model: {e}")
        evaluation = {}
    
    # Export results
    try:
        export_forecast_csv(results, args.output)
        print(f"\nResults exported to: {args.output}")
    except Exception as e:
        print(f"Warning: Could not export results: {e}")
    
    # Generate report
    try:
        summary = forecaster.get_data_summary()
        report = create_forecast_report(results, summary, evaluation)
        
        report_file = args.output.replace('.csv', '_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Warning: Could not generate report: {e}")
    
    print("\nForecasting completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())