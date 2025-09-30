# Forecast Toolkit

A Python toolkit for intelligent time series forecasting using AI agents. This toolkit adapts forecasting algorithms dynamically based on data characteristics and contextual information, providing more flexible and accurate predictions than traditional static approaches.

## 🌟 Key Features

- **AI-Powered Algorithm Selection**: Uses OpenAI agents to automatically select the best forecasting algorithm based on your data
- **Contextual Forecasting**: Incorporates business context and explanations to improve forecast accuracy
- **Adaptive Models**: Automatically adjusts algorithms and parameters based on changing data patterns
- **Multiple Algorithm Support**: Linear regression, ARIMA, exponential smoothing, Prophet, Random Forest, and neural networks
- **Data Quality Assessment**: Built-in data validation and quality scoring
- **Comprehensive Evaluation**: Cross-validation and multiple performance metrics
- **Easy Integration**: Simple API with pandas DataFrame support

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from forecast import Forecaster
from forecast.utils.helpers import generate_sample_data

# Generate sample data
data = generate_sample_data(length=100, trend=0.1, seasonality=True)

# Create forecaster
forecaster = Forecaster()

# Load and prepare data
forecaster.load_data(data)
forecaster.prepare_data()

# Fit with context
context = """
Sales data showing growth trend with seasonal patterns.
Expecting continued growth due to new product launch.
"""

forecaster.fit(context=context)

# Generate forecasts
results = forecaster.predict(horizon=14)

print(f"Next 14-day forecasts: {results['forecasts']}")
```

### With OpenAI Integration

```python
import os
from forecast import Forecaster, ForecastConfig

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Create forecaster with AI agent
config = ForecastConfig(
    openai_model="gpt-4",
    max_forecast_horizon=30
)
forecaster = Forecaster(config)

# The AI agent will automatically:
# 1. Analyze your data characteristics
# 2. Select the best algorithm
# 3. Optimize parameters
# 4. Interpret business context
```

## 📊 Data Formats Supported

The toolkit accepts multiple data formats:

```python
# From pandas DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'sales': np.random.randn(100)
})
forecaster.load_data(df, time_column='date', value_column='sales')

# From dictionary
data = {
    'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'value': [100, 110, 105]
}
forecaster.load_data(data)

# From list (assumes daily frequency)
values = [1, 2, 3, 4, 5, 6, 7]
forecaster.load_data(values)
```

## 🤖 AI Agent Capabilities

When OpenAI integration is enabled, the AI agent provides:

### Algorithm Selection
- Analyzes data length, trend, seasonality, and stationarity
- Considers forecast horizon and business context
- Selects optimal algorithm with confidence scoring

### Parameter Optimization
- Optimizes algorithm-specific parameters
- Adapts to data characteristics
- Provides reasoning for parameter choices

### Context Interpretation
- Extracts key factors from business context
- Identifies positive/negative impacts
- Suggests forecast adjustments

## 📈 Supported Algorithms

| Algorithm | Best For | Key Features |
|-----------|----------|--------------|
| Linear Regression | Simple trends | Fast, interpretable |
| ARIMA | Stationary series | Handles autocorrelation |
| Exponential Smoothing | Trending data | Adaptive to recent changes |
| Prophet | Strong seasonality | Handles holidays and events |
| Random Forest | Complex patterns | Non-linear relationships |
| Neural Network | Large datasets | Deep pattern recognition |

## 🛠 Configuration Options

```python
from forecast import ForecastConfig

config = ForecastConfig(
    # OpenAI settings
    openai_api_key="your-key",
    openai_model="gpt-4",
    temperature=0.1,
    
    # Forecasting parameters
    max_forecast_horizon=60,
    confidence_level=0.95,
    min_data_points=10,
    
    # Validation settings
    enable_cross_validation=True,
    cv_folds=5,
    
    # Available algorithms
    available_algorithms=[
        "linear_regression",
        "arima", 
        "exponential_smoothing",
        "prophet",
        "random_forest",
        "neural_network"
    ]
)
```

## 📊 Data Quality Assessment

```python
from forecast.utils.helpers import validate_data_quality

# Check data quality
quality = validate_data_quality(your_data)

print(f"Quality Score: {quality['quality_score']}/100")
print(f"Assessment: {quality['assessment']}")
print(f"Missing values: {quality['missing_count']}")
print(f"Outliers: {quality['outlier_count']}")
```

## 📋 Complete Workflow Example

```python
from forecast import Forecaster
from forecast.utils.helpers import (
    plot_forecast, 
    create_forecast_report,
    export_forecast_csv
)

# 1. Create and configure forecaster
forecaster = Forecaster()

# 2. Load your data
forecaster.load_data(your_data, time_column='date', value_column='sales')

# 3. Prepare and clean data
forecaster.prepare_data(
    handle_missing='interpolate',
    remove_outliers=True
)

# 4. Fit with business context
context = """
Retail sales data with seasonal patterns. 
Recent marketing campaign increased sales.
Holiday season approaching - expect higher demand.
"""

forecaster.fit(context=context)

# 5. Generate forecasts
results = forecaster.predict(horizon=30)

# 6. Evaluate performance
evaluation = forecaster.evaluate()
print(f"Model accuracy (MAE): {evaluation['mae']:.2f}")

# 7. Create visualizations and reports
plot_forecast(results, your_data)
report = create_forecast_report(results, forecaster.get_data_summary(), evaluation)
export_forecast_csv(results, "forecasts.csv")
```

## 🧪 Testing

Run the test suite:

```bash
python tests/test_basic.py
```

Or using pytest:

```bash
pytest tests/
```

## 📖 Examples

Check out the `examples/` directory for detailed usage examples:

- `basic_usage.py`: Complete workflow demonstration
- Algorithm comparison examples
- Custom data format examples
- Advanced configuration examples

## 🔧 Troubleshooting

### Common Issues

**"OpenAI API key is required"**
- Set the `OPENAI_API_KEY` environment variable
- Or pass `openai_api_key=None` to disable AI features

**"Insufficient data"**
- Ensure you have at least 10 data points
- Check for missing values in your dataset

**"No numeric column found"**
- Specify the correct `value_column` when loading data
- Ensure your value column contains numeric data

### Without OpenAI API

The toolkit works perfectly without OpenAI integration:

```python
from forecast import Forecaster, ForecastConfig

# Disable OpenAI integration
config = ForecastConfig(openai_api_key=None)
forecaster = Forecaster(config)

# Uses rule-based algorithm selection and default parameters
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🏗 Architecture

```
forecast/
├── core/           # Main forecasting logic
│   ├── forecaster.py    # Main Forecaster class
│   └── config.py        # Configuration management
├── agents/         # AI agent integration
│   └── openai_agent.py  # OpenAI-powered agent
├── data/           # Data processing
│   └── processor.py     # Time series processing
├── models/         # Forecasting models
│   └── adaptive.py      # Adaptive model wrapper
└── utils/          # Utility functions
    └── helpers.py       # Helper functions and plotting
```

## 🎯 Roadmap

- [ ] Support for additional algorithms (XGBoost, LightGBM)
- [ ] Real-time forecasting capabilities
- [ ] Web dashboard interface
- [ ] Integration with cloud data sources
- [ ] Automated model retraining
- [ ] Ensemble forecasting methods

---

**Get started with intelligent forecasting today!** 🚀