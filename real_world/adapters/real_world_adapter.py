"""
Real-world API Adapter
Gerçek dünya API'leri için adapter
"""

import requests
import json
import numpy as np
import time
import base64
import io
from PIL import Image
from typing import Dict, List, Optional, Union
import logging


class RealWorldAPIAdapter:
    """
    Gerçek dünya API'leri için adapter sınıfı
    Mevcut VictimModelAPI interface'ini implement eder
    """
    
    def __init__(self, 
                 api_endpoint: str,
                 api_key: Optional[str] = None,
                 input_format: str = "base64_json",
                 rate_limit_delay: float = 0.5,
                 max_retries: int = 3):
        
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.input_format = input_format
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Setup headers
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        # Statistics
        self.request_count = 0
        self.failed_requests = 0
        self.total_cost = 0.0  # If API charges per request
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def predict(self, image_data: Union[np.ndarray, str], 
                return_probabilities: bool = True) -> Dict:
        """
        Ana tahmin fonksiyonu - VictimModelAPI interface'ini taklit eder
        """
        try:
            # Convert input to required format
            formatted_data = self._format_input(image_data)
            
            # Make API request with retries
            response = self._make_request_with_retry(formatted_data)
            
            # Parse and standardize response
            standardized_response = self._parse_response(response, return_probabilities)
            
            # Update statistics
            self.request_count += 1
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return standardized_response
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Prediction failed: {str(e)}")
            
            # Return dummy response to avoid breaking the extraction process
            return {
                'status': 'error',
                'predictions': [0],  # Default class
                'probabilities': [[0.1] * 10] if return_probabilities else None,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            'model_type': 'external_api',
            'api_endpoint': self.api_endpoint,
            'request_count': self.request_count,
            'failed_requests': self.failed_requests,
            'success_rate': (self.request_count - self.failed_requests) / max(self.request_count, 1),
            'estimated_cost': self.total_cost
        }
    
    def _format_input(self, image_data: Union[np.ndarray, str]) -> Dict:
        """Girdiyi API formatına çevir"""
        
        if self.input_format == "base64_json":
            if isinstance(image_data, np.ndarray):
                # Convert numpy array to PIL Image
                if image_data.dtype != np.uint8:
                    image_data = (image_data * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(image_data)
                
                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return {
                    'image': img_str,
                    'format': 'base64'
                }
            
            elif isinstance(image_data, str):
                # Assume it's already base64 encoded
                return {
                    'image': image_data,
                    'format': 'base64'
                }
        
        elif self.input_format == "url":
            # Some APIs accept image URLs
            return {
                'image_url': image_data,
                'format': 'url'
            }
        
        elif self.input_format == "raw_array":
            # Send as raw numpy array
            if isinstance(image_data, np.ndarray):
                return {
                    'image_array': image_data.tolist(),
                    'shape': image_data.shape,
                    'format': 'array'
                }
        
        # Default fallback
        return {'data': image_data}
    
    def _make_request_with_retry(self, data: Dict) -> requests.Response:
        """Retry logic ile API isteği yap"""
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=data,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Request failed (attempt {attempt+1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
        
        # All retries failed
        raise last_exception or Exception("All retry attempts failed")
    
    def _parse_response(self, response: requests.Response, 
                       return_probabilities: bool) -> Dict:
        """API yanıtını standardize et"""
        
        try:
            data = response.json()
            
            # Different APIs have different response formats
            # Try to extract predictions and probabilities
            
            predictions = None
            probabilities = None
            
            # Common patterns for different APIs
            if 'predictions' in data:
                predictions = data['predictions']
            elif 'classes' in data:
                predictions = data['classes']
            elif 'results' in data:
                predictions = data['results']
            elif 'label' in data:
                predictions = [data['label']]
            elif 'class_id' in data:
                predictions = [data['class_id']]
            
            # Extract probabilities/confidence scores
            if 'probabilities' in data:
                probabilities = data['probabilities']
            elif 'scores' in data:
                probabilities = data['scores']
            elif 'confidence' in data:
                conf = data['confidence']
                if isinstance(conf, (int, float)):
                    # Single confidence score, create probability distribution
                    probabilities = [[conf, 1-conf]]  # Binary case
                else:
                    probabilities = [conf]
            
            # Ensure predictions is a list of integers
            if predictions is not None:
                if not isinstance(predictions, list):
                    predictions = [predictions]
                
                # Convert to integers if possible
                try:
                    predictions = [int(p) for p in predictions]
                except (ValueError, TypeError):
                    # If string labels, map to integers
                    unique_labels = list(set(predictions))
                    label_map = {label: i for i, label in enumerate(unique_labels)}
                    predictions = [label_map[p] for p in predictions]
            
            # Create standardized response
            standardized = {
                'status': 'success',
                'predictions': predictions or [0],
            }
            
            if return_probabilities and probabilities is not None:
                standardized['probabilities'] = probabilities
            
            return standardized
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse API response: {str(e)}")
            
            # Return dummy response
            return {
                'status': 'error',
                'predictions': [0],
                'probabilities': [[0.1] * 10] if return_probabilities else None,
                'error': f"Parse error: {str(e)}"
            }


class PopularAPIAdapters:
    """
    Popüler API servisleri için hazır adapter'lar
    """
    
    @staticmethod
    def google_vision_api(api_key: str) -> RealWorldAPIAdapter:
        """Google Vision API adapter"""
        return RealWorldAPIAdapter(
            api_endpoint="https://vision.googleapis.com/v1/images:annotate",
            api_key=api_key,
            input_format="base64_json",
            rate_limit_delay=0.1  # Google is usually fast
        )
    
    @staticmethod
    def aws_rekognition(access_key: str, secret_key: str, region: str = "us-east-1") -> RealWorldAPIAdapter:
        """AWS Rekognition adapter (simplified)"""
        # Note: Real AWS implementation would need proper signing
        return RealWorldAPIAdapter(
            api_endpoint=f"https://rekognition.{region}.amazonaws.com/",
            api_key=access_key,  # Simplified - real AWS needs proper auth
            input_format="base64_json",
            rate_limit_delay=0.2
        )
    
    @staticmethod
    def azure_computer_vision(api_key: str, endpoint: str) -> RealWorldAPIAdapter:
        """Azure Computer Vision adapter"""
        return RealWorldAPIAdapter(
            api_endpoint=f"{endpoint}/vision/v3.2/analyze",
            api_key=api_key,
            input_format="base64_json",
            rate_limit_delay=0.1
        )
    
    @staticmethod
    def custom_api(endpoint: str, api_key: str = None, 
                  input_format: str = "base64_json",
                  rate_limit: float = 0.5) -> RealWorldAPIAdapter:
        """Custom API adapter"""
        return RealWorldAPIAdapter(
            api_endpoint=endpoint,
            api_key=api_key,
            input_format=input_format,
            rate_limit_delay=rate_limit
        )


# Example configuration for real-world usage
def setup_real_world_attack(api_config: Dict) -> RealWorldAPIAdapter:
    """
    Gerçek dünya saldırısı için setup
    
    Args:
        api_config: {
            'type': 'google_vision' | 'aws_rekognition' | 'azure_cv' | 'custom',
            'api_key': 'your-api-key',
            'endpoint': 'custom-endpoint-if-needed',
            'rate_limit': 0.5,  # seconds between requests
            'input_format': 'base64_json'
        }
    """
    
    api_type = api_config.get('type', 'custom')
    
    if api_type == 'google_vision':
        return PopularAPIAdapters.google_vision_api(api_config['api_key'])
    elif api_type == 'aws_rekognition':
        return PopularAPIAdapters.aws_rekognition(
            api_config['api_key'], 
            api_config.get('secret_key'),
            api_config.get('region', 'us-east-1')
        )
    elif api_type == 'azure_cv':
        return PopularAPIAdapters.azure_computer_vision(
            api_config['api_key'],
            api_config['endpoint']
        )
    else:  # custom
        return PopularAPIAdapters.custom_api(
            api_config['endpoint'],
            api_config.get('api_key'),
            api_config.get('input_format', 'base64_json'),
            api_config.get('rate_limit', 0.5)
        )


if __name__ == "__main__":
    # Example usage
    print("Real-world API adapter ready!")
    print("Use this in your extraction attacks by replacing VictimModelAPI")
