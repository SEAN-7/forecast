"""Configuration settings for the forecast toolkit."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class ForecastConfig(BaseModel):
    """Configuration for forecasting operations."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key for agent operations"
    )
    openai_model: str = Field(
        default="gpt-4",
        description="OpenAI model to use for forecasting agent"
    )
    
    # Forecasting Parameters
    max_forecast_horizon: int = Field(
        default=30,
        description="Maximum number of periods to forecast ahead"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for prediction intervals"
    )
    
    # Data Processing
    min_data_points: int = Field(
        default=10,
        description="Minimum number of data points required for forecasting"
    )
    max_data_points: int = Field(
        default=10000,
        description="Maximum number of data points to use for forecasting"
    )
    
    # Agent Parameters
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for agent operations"
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for OpenAI model responses"
    )
    
    # Algorithm Selection
    available_algorithms: list = Field(
        default_factory=lambda: [
            "linear_regression",
            "arima",
            "exponential_smoothing",
            "prophet",
            "random_forest",
            "neural_network"
        ],
        description="List of available forecasting algorithms"
    )
    
    # Validation
    enable_cross_validation: bool = Field(
        default=True,
        description="Enable cross-validation for model selection"
    )
    cv_folds: int = Field(
        default=5,
        description="Number of folds for cross-validation"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "FORECAST_"
        case_sensitive = False


# Default configuration instance
default_config = ForecastConfig()