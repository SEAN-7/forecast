"""
Forecast - A Python toolkit for forecasting using OpenAI agents.

This package provides adaptive forecasting capabilities by leveraging AI agents
that can dynamically select and adjust algorithms based on time series data
and contextual information.
"""

__version__ = "0.1.0"
__author__ = "SEAN-7"

from .core.forecaster import Forecaster
from .core.config import ForecastConfig
from .data.processor import TimeSeriesProcessor
from .models.adaptive import AdaptiveModel

__all__ = ["Forecaster", "ForecastConfig", "TimeSeriesProcessor", "AdaptiveModel"]