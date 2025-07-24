"""
Real-world Target Analysis
Gerçek hedef model analizi
"""

import requests
import json
import numpy as np
from typing import Dict, List, Optional
import time
import base64
from PIL import Image
import io


class RealWorldTargetAnalyzer:
    """
    Gerçek dünya hedef model analizi
    """
    
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
            # or self.headers['X-API-Key'] = api_key  # depending on API format
        
        # Analysis results
        self.analysis_results = {}
        
    def probe_api_structure(self) -> Dict:
        """API yapısını keşfet"""
        print("🔍 Probing API structure...")
        
        results = {
            'endpoints': [],
            'input_format': None,
            'output_format': None,
            'rate_limits': None,
            'error_patterns': []
        }
        
        # Test different endpoints
        common_endpoints = [
            '/predict', '/inference', '/classify', '/detect',
            '/v1/predict', '/api/predict', '/model/predict'
        ]
        
        for endpoint in common_endpoints:
            test_url = self.api_endpoint.rstrip('/') + endpoint
            try:
                response = requests.get(test_url, headers=self.headers, timeout=5)
                if response.status_code != 404:
                    results['endpoints'].append({
                        'endpoint': endpoint,
                        'status': response.status_code,
                        'response': response.text[:200]
                    })
            except Exception as e:
                continue
        
        return results
    
    def test_input_formats(self) -> Dict:
        """Giriş formatlarını test et"""
        print("📥 Testing input formats...")
        
        # Create test images in different formats
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        formats_to_test = {
            'base64_json': self._encode_base64_json(pil_image),
            'form_data': self._encode_form_data(pil_image),
            'raw_bytes': self._encode_raw_bytes(pil_image),
            'numpy_array': test_image.tolist(),
            'url_reference': 'https://example.com/test.jpg'
        }
        
        results = {}
        for format_name, data in formats_to_test.items():
            try:
                response = self._send_test_request(data, format_name)
                results[format_name] = {
                    'status': response.status_code,
                    'success': response.status_code == 200,
                    'response_preview': response.text[:200]
                }
            except Exception as e:
                results[format_name] = {
                    'status': 'error',
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def analyze_rate_limits(self) -> Dict:
        """Rate limit'leri analiz et"""
        print("⏱️ Analyzing rate limits...")
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_data = self._encode_base64_json(Image.fromarray(test_image))
        
        request_times = []
        rate_limit_hit = False
        
        for i in range(20):  # Test with 20 rapid requests
            start_time = time.time()
            try:
                response = self._send_test_request(test_data, 'base64_json')
                end_time = time.time()
                
                request_times.append(end_time - start_time)
                
                if response.status_code == 429:  # Too Many Requests
                    rate_limit_hit = True
                    break
                    
                time.sleep(0.1)  # Small delay between requests
                
            except Exception as e:
                break
        
        return {
            'rate_limit_detected': rate_limit_hit,
            'avg_response_time': np.mean(request_times) if request_times else None,
            'requests_tested': len(request_times),
            'recommendation': self._get_rate_limit_recommendation(request_times, rate_limit_hit)
        }
    
    def detect_model_properties(self) -> Dict:
        """Model özelliklerini tespit et"""
        print("🔬 Detecting model properties...")
        
        results = {
            'input_size': None,
            'num_classes': None,
            'confidence_scores': False,
            'class_names': [],
            'preprocessing_hints': []
        }
        
        # Test with different image sizes
        sizes_to_test = [(224, 224), (299, 299), (512, 512), (640, 640)]
        
        for size in sizes_to_test:
            test_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            test_data = self._encode_base64_json(Image.fromarray(test_image))
            
            try:
                response = self._send_test_request(test_data, 'base64_json')
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Analyze response structure
                    if 'predictions' in response_data:
                        predictions = response_data['predictions']
                        if isinstance(predictions, list) and len(predictions) > 0:
                            results['num_classes'] = len(predictions)
                            results['input_size'] = size
                            
                            # Check for confidence scores
                            if all(isinstance(p, (int, float)) for p in predictions):
                                results['confidence_scores'] = True
                    
                    break
                    
            except Exception as e:
                continue
        
        return results
    
    def _encode_base64_json(self, image: Image.Image) -> Dict:
        """Base64 JSON formatında encode et"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'image': img_str,
            'format': 'base64'
        }
    
    def _encode_form_data(self, image: Image.Image) -> Dict:
        """Form data formatında encode et"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        
        return {
            'files': {'image': ('test.jpg', buffer.getvalue(), 'image/jpeg')}
        }
    
    def _encode_raw_bytes(self, image: Image.Image) -> bytes:
        """Raw bytes formatında encode et"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def _send_test_request(self, data, format_type: str):
        """Test isteği gönder"""
        if format_type == 'form_data':
            return requests.post(
                self.api_endpoint,
                files=data['files'],
                headers=self.headers,
                timeout=10
            )
        else:
            return requests.post(
                self.api_endpoint,
                json=data if isinstance(data, dict) else {'data': data},
                headers=self.headers,
                timeout=10
            )
    
    def _get_rate_limit_recommendation(self, times: List[float], hit_limit: bool) -> str:
        """Rate limit önerisi"""
        if hit_limit:
            return "Rate limit detected. Use delays of 1-2 seconds between requests."
        elif np.mean(times) > 2.0:
            return "Slow API detected. Consider using batch requests if available."
        else:
            return "Fast API. Can use aggressive querying (0.1-0.5s delays)."


def analyze_target_api(api_endpoint: str, api_key: Optional[str] = None) -> Dict:
    """Hedef API'yi comprehensive analiz et"""
    
    print(f"🎯 Analyzing target API: {api_endpoint}")
    print("=" * 60)
    
    analyzer = RealWorldTargetAnalyzer(api_endpoint, api_key)
    
    # Full analysis
    analysis = {
        'api_structure': analyzer.probe_api_structure(),
        'input_formats': analyzer.test_input_formats(),
        'rate_limits': analyzer.analyze_rate_limits(),
        'model_properties': analyzer.detect_model_properties(),
        'timestamp': time.time()
    }
    
    # Print summary
    print("\n📋 Analysis Summary:")
    print(f"✅ Available endpoints: {len(analysis['api_structure']['endpoints'])}")
    
    successful_formats = [k for k, v in analysis['input_formats'].items() if v['success']]
    print(f"✅ Working input formats: {successful_formats}")
    
    if analysis['model_properties']['num_classes']:
        print(f"✅ Detected classes: {analysis['model_properties']['num_classes']}")
    
    if analysis['model_properties']['input_size']:
        print(f"✅ Input size: {analysis['model_properties']['input_size']}")
    
    print(f"⚠️ Rate limit recommendation: {analysis['rate_limits']['recommendation']}")
    
    return analysis


# Example usage
if __name__ == "__main__":
    # Example: Analyze a public vision API
    # analysis = analyze_target_api(
    #     api_endpoint="https://api.example.com/v1/vision/classify",
    #     api_key="your-api-key-here"
    # )
    
    print("Use this script to analyze your target API before launching extraction attack.")
