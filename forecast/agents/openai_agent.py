"""OpenAI agent for adaptive forecasting algorithm selection."""

import json
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from ..core.config import ForecastConfig

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class ForecastingAgent:
    """OpenAI-powered agent for selecting and configuring forecasting algorithms."""
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize the forecasting agent.
        
        Args:
            config: Configuration object
        """
        if not OPENAI_AVAILABLE:
            raise ValueError(
                "OpenAI package is not available. Install it with: pip install openai>=1.0.0"
            )
            
        self.config = config or ForecastConfig()
        
        if not self.config.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide it in the config."
            )
        
        self.client = OpenAI(api_key=self.config.openai_api_key)
    
    def select_algorithm(
        self,
        time_series_features: Dict[str, Any],
        context: str = "",
        forecast_horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Select the best forecasting algorithm based on data characteristics and context.
        
        Args:
            time_series_features: Features extracted from the time series
            context: Contextual information about the forecasting task
            forecast_horizon: Number of periods to forecast ahead
            
        Returns:
            Dictionary with selected algorithm and configuration
        """
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_selection_prompt(
            time_series_features, context, forecast_horizon
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=1000
            )
            
            result = self._parse_algorithm_response(response.choices[0].message.content)
            return result
            
        except Exception as e:
            # Fallback to simple rule-based selection
            return self._fallback_algorithm_selection(time_series_features)
    
    def optimize_parameters(
        self,
        algorithm: str,
        time_series_features: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Optimize parameters for the selected algorithm.
        
        Args:
            algorithm: Selected forecasting algorithm
            time_series_features: Features extracted from the time series
            context: Contextual information
            
        Returns:
            Dictionary with optimized parameters
        """
        system_prompt = self._create_optimization_prompt()
        user_prompt = self._create_parameter_prompt(
            algorithm, time_series_features, context
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=800
            )
            
            result = self._parse_parameter_response(
                response.choices[0].message.content, algorithm
            )
            return result
            
        except Exception as e:
            # Fallback to default parameters
            return self._get_default_parameters(algorithm)
    
    def interpret_context(self, context: str) -> Dict[str, Any]:
        """
        Interpret contextual information to extract relevant factors for forecasting.
        
        Args:
            context: Free-text description of the context/situation
            
        Returns:
            Dictionary with interpreted context factors
        """
        if not context.strip():
            return {"factors": [], "impact": "unknown", "adjustments": []}
        
        system_prompt = self._create_context_prompt()
        user_prompt = f"""
        Analyze the following context for forecasting:
        
        Context: {context}
        
        Extract key factors that might impact the forecast and suggest adjustments.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=600
            )
            
            result = self._parse_context_response(response.choices[0].message.content)
            return result
            
        except Exception as e:
            return {"factors": [], "impact": "unknown", "adjustments": []}
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for algorithm selection."""
        return f"""
        You are an expert forecasting consultant with deep knowledge of time series analysis.
        Your task is to select the best forecasting algorithm based on data characteristics.
        
        Available algorithms: {', '.join(self.config.available_algorithms)}
        
        Consider these factors when selecting:
        - Data length and quality
        - Trend and seasonality patterns
        - Stationarity
        - Forecast horizon
        - Context and external factors
        
        Always respond with a JSON object containing:
        {{
            "algorithm": "selected_algorithm_name",
            "confidence": 0.0-1.0,
            "reasoning": "explanation of choice",
            "alternative": "backup_algorithm_name"
        }}
        """
    
    def _create_optimization_prompt(self) -> str:
        """Create the system prompt for parameter optimization."""
        return """
        You are an expert in time series forecasting parameter optimization.
        Your task is to suggest optimal parameters for the given algorithm and data characteristics.
        
        Always respond with a JSON object containing the parameter suggestions:
        {
            "parameters": {
                "param1": value1,
                "param2": value2,
                ...
            },
            "confidence": 0.0-1.0,
            "reasoning": "explanation of parameter choices"
        }
        """
    
    def _create_context_prompt(self) -> str:
        """Create the system prompt for context interpretation."""
        return """
        You are an expert in interpreting business and economic context for forecasting.
        Your task is to extract key factors from contextual information that might impact forecasts.
        
        Always respond with a JSON object:
        {
            "factors": ["factor1", "factor2", ...],
            "impact": "positive/negative/neutral/mixed",
            "adjustments": ["adjustment1", "adjustment2", ...],
            "confidence": 0.0-1.0
        }
        """
    
    def _create_selection_prompt(
        self,
        features: Dict[str, Any],
        context: str,
        horizon: int
    ) -> str:
        """Create the user prompt for algorithm selection."""
        return f"""
        Please select the best forecasting algorithm for this time series:
        
        Data Characteristics:
        - Length: {features.get('length', 'unknown')}
        - Trend: {features.get('trend', 'unknown')}
        - Seasonality: {features.get('seasonality', 'unknown')}
        - Stationarity: {features.get('stationarity', 'unknown')}
        - Frequency: {features.get('frequency', 'unknown')}
        - Missing data ratio: {features.get('missing_ratio', 0)}
        - Autocorrelation: {features.get('autocorrelation', [])}
        
        Context: {context if context else 'No additional context provided'}
        
        Forecast Horizon: {horizon} periods ahead
        
        Select the most appropriate algorithm and explain your reasoning.
        """
    
    def _create_parameter_prompt(
        self,
        algorithm: str,
        features: Dict[str, Any],
        context: str
    ) -> str:
        """Create the user prompt for parameter optimization."""
        return f"""
        Optimize parameters for the {algorithm} algorithm given these characteristics:
        
        Data Characteristics:
        - Length: {features.get('length', 'unknown')}
        - Trend: {features.get('trend', 'unknown')}
        - Seasonality: {features.get('seasonality', 'unknown')}
        - Stationarity: {features.get('stationarity', 'unknown')}
        - Frequency: {features.get('frequency', 'unknown')}
        
        Context: {context if context else 'No additional context provided'}
        
        Suggest optimal parameters for this algorithm and data combination.
        """
    
    def _parse_algorithm_response(self, response: str) -> Dict[str, Any]:
        """Parse the algorithm selection response."""
        try:
            # Try to extract JSON from the response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                
                # Validate required fields
                if 'algorithm' in result and result['algorithm'] in self.config.available_algorithms:
                    return {
                        'algorithm': result['algorithm'],
                        'confidence': result.get('confidence', 0.5),
                        'reasoning': result.get('reasoning', 'No reasoning provided'),
                        'alternative': result.get('alternative', 'linear_regression')
                    }
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback parsing
        for algorithm in self.config.available_algorithms:
            if algorithm.lower() in response.lower():
                return {
                    'algorithm': algorithm,
                    'confidence': 0.5,
                    'reasoning': 'Parsed from text response',
                    'alternative': 'linear_regression'
                }
        
        # Final fallback
        return {
            'algorithm': 'linear_regression',
            'confidence': 0.3,
            'reasoning': 'Default fallback algorithm',
            'alternative': 'arima'
        }
    
    def _parse_parameter_response(self, response: str, algorithm: str) -> Dict[str, Any]:
        """Parse the parameter optimization response."""
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                
                return {
                    'parameters': result.get('parameters', {}),
                    'confidence': result.get('confidence', 0.5),
                    'reasoning': result.get('reasoning', 'No reasoning provided')
                }
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback to default parameters
        return self._get_default_parameters(algorithm)
    
    def _parse_context_response(self, response: str) -> Dict[str, Any]:
        """Parse the context interpretation response."""
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                
                return {
                    'factors': result.get('factors', []),
                    'impact': result.get('impact', 'unknown'),
                    'adjustments': result.get('adjustments', []),
                    'confidence': result.get('confidence', 0.5)
                }
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {"factors": [], "impact": "unknown", "adjustments": [], "confidence": 0.0}
    
    def _fallback_algorithm_selection(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based algorithm selection."""
        length = features.get('length', 0)
        seasonality = features.get('seasonality', {})
        
        if length < 20:
            algorithm = 'linear_regression'
        elif seasonality.get('has_seasonality', False):
            algorithm = 'exponential_smoothing'
        elif length > 100:
            algorithm = 'arima'
        else:
            algorithm = 'linear_regression'
        
        return {
            'algorithm': algorithm,
            'confidence': 0.4,
            'reasoning': 'Rule-based fallback selection',
            'alternative': 'linear_regression'
        }
    
    def _get_default_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm."""
        defaults = {
            'linear_regression': {
                'parameters': {'fit_intercept': True},
                'confidence': 0.5,
                'reasoning': 'Default parameters'
            },
            'arima': {
                'parameters': {'order': (1, 1, 1), 'seasonal_order': (0, 0, 0, 0)},
                'confidence': 0.5,
                'reasoning': 'Default parameters'
            },
            'exponential_smoothing': {
                'parameters': {'trend': 'add', 'seasonal': 'add'},
                'confidence': 0.5,
                'reasoning': 'Default parameters'
            },
            'prophet': {
                'parameters': {'seasonality_mode': 'additive'},
                'confidence': 0.5,
                'reasoning': 'Default parameters'
            },
            'random_forest': {
                'parameters': {'n_estimators': 100, 'max_depth': 10},
                'confidence': 0.5,
                'reasoning': 'Default parameters'
            },
            'neural_network': {
                'parameters': {'hidden_size': 50, 'num_layers': 2},
                'confidence': 0.5,
                'reasoning': 'Default parameters'
            }
        }
        
        return defaults.get(algorithm, {
            'parameters': {},
            'confidence': 0.5,
            'reasoning': 'Default parameters'
        })