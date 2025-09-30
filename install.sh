#!/bin/bash

# Forecast Toolkit Installation Script

echo "Installing Forecast Toolkit..."

# Install core dependencies
echo "Installing core dependencies..."
pip install pandas numpy scikit-learn matplotlib seaborn pydantic python-dotenv

# Install optional OpenAI dependency
echo "Installing optional OpenAI dependency..."
pip install openai || echo "OpenAI installation failed - toolkit will work without AI agent"

# Verify installation
echo "Verifying installation..."
python -c "
import sys
sys.path.insert(0, '.')
from forecast import Forecaster
from forecast.utils.helpers import generate_sample_data

print('Testing basic functionality...')
data = generate_sample_data(20)
forecaster = Forecaster()
forecaster.load_data(data)
forecaster.prepare_data()
forecaster.fit()
results = forecaster.predict(horizon=3)
print(f'✓ Installation successful! Generated {len(results[\"forecasts\"])} predictions.')
"

echo "Installation complete!"
echo ""
echo "Usage:"
echo "  from forecast import Forecaster"
echo "  forecaster = Forecaster()"
echo ""
echo "For examples, run: python examples/basic_usage.py"