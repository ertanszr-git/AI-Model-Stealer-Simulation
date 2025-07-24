"""
API-Specific Attack Examples
Popüler AI API'lerine özel saldırı örnekleri
"""

import sys
import os
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import base64
import json
import time
from PIL import Image
import io
import requests
from typing import Dict, Any, List


class GoogleVisionAttack:
    """Google Cloud Vision API için özelleştirilmiş saldırı"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        self.headers = {"Content-Type": "application/json"}
        
    def image_to_base64(self, image: np.ndarray) -> str:
        """NumPy array'i base64'e çevir"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def create_request_payload(self, image: np.ndarray) -> Dict:
        """Google Vision API request formatı"""
        base64_image = self.image_to_base64(image)
        
        return {
            "requests": [
                {
                    "image": {
                        "content": base64_image
                    },
                    "features": [
                        {
                            "type": "LABEL_DETECTION",
                            "maxResults": 10
                        },
                        {
                            "type": "OBJECT_LOCALIZATION",
                            "maxResults": 10
                        }
                    ]
                }
            ]
        }
    
    def extract_predictions(self, response: Dict) -> List[str]:
        """API yanıtından tahminleri çıkar"""
        predictions = []
        
        if 'responses' in response and len(response['responses']) > 0:
            resp = response['responses'][0]
            
            # Label detection
            if 'labelAnnotations' in resp:
                for label in resp['labelAnnotations']:
                    predictions.append({
                        'label': label['description'],
                        'confidence': label['score'],
                        'type': 'label'
                    })
            
            # Object detection
            if 'localizedObjectAnnotations' in resp:
                for obj in resp['localizedObjectAnnotations']:
                    predictions.append({
                        'label': obj['name'],
                        'confidence': obj['score'],
                        'type': 'object'
                    })
        
        return predictions
    
    def query_api(self, image: np.ndarray) -> Dict:
        """Single API query"""
        payload = self.create_request_payload(image)
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                api_response = response.json()
                predictions = self.extract_predictions(api_response)
                
                return {
                    'status': 'success',
                    'predictions': predictions,
                    'raw_response': api_response
                }
            else:
                return {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def batch_attack(self, num_queries: int = 100) -> Dict:
        """Batch saldırı Google Vision API'ye"""
        print(f"🎯 Starting Google Vision API attack with {num_queries} queries...")
        
        results = {
            'successful_queries': 0,
            'failed_queries': 0,
            'predictions': [],
            'query_images': [],
            'costs': 0.0
        }
        
        for i in range(num_queries):
            # Rastgele görüntü oluştur
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # API'yi sorgula
            response = self.query_api(image)
            
            if response['status'] == 'success':
                results['successful_queries'] += 1
                results['predictions'].append(response['predictions'])
                results['query_images'].append(image)
            else:
                results['failed_queries'] += 1
                print(f"Query {i+1} failed: {response['error']}")
            
            # Maliyet hesapla (Google Vision: $1.50 per 1000 requests)
            results['costs'] += 0.0015
            
            # Rate limiting
            time.sleep(0.5)
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{num_queries} queries completed")
        
        success_rate = results['successful_queries'] / num_queries
        print(f"✅ Attack completed! Success rate: {success_rate:.2%}")
        print(f"💰 Total cost: ${results['costs']:.2f}")
        
        return results


class AWSRekognitionAttack:
    """AWS Rekognition için özelleştirilmiş saldırı"""
    
    def __init__(self, aws_access_key: str, aws_secret_key: str, region: str = 'us-east-1'):
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region = region
        
        # boto3 import (optional)
        try:
            import boto3
            self.rekognition = boto3.client(
                'rekognition',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            self.boto3_available = True
        except ImportError:
            print("⚠️ boto3 not available, using REST API")
            self.boto3_available = False
            self.endpoint = f"https://rekognition.{region}.amazonaws.com/"
    
    def query_with_boto3(self, image: np.ndarray) -> Dict:
        """boto3 ile sorgulama"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # PIL formatına çevir
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        try:
            # Label detection
            labels_response = self.rekognition.detect_labels(
                Image={'Bytes': image_bytes},
                MaxLabels=10,
                MinConfidence=50
            )
            
            # Object detection
            objects_response = self.rekognition.detect_custom_labels(
                Image={'Bytes': image_bytes},
                ProjectVersionArn='arn:aws:rekognition:us-east-1:123456789012:project/my-project/version/my-version/1234567890123'
            )
            
            predictions = []
            
            # Labels
            for label in labels_response['Labels']:
                predictions.append({
                    'label': label['Name'],
                    'confidence': label['Confidence'] / 100.0,
                    'type': 'label'
                })
            
            return {
                'status': 'success',
                'predictions': predictions
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def batch_attack(self, num_queries: int = 100) -> Dict:
        """AWS Rekognition batch saldırı"""
        if not self.boto3_available:
            print("❌ boto3 required for AWS Rekognition attack")
            return {}
        
        print(f"🎯 Starting AWS Rekognition attack with {num_queries} queries...")
        
        results = {
            'successful_queries': 0,
            'failed_queries': 0,
            'predictions': [],
            'costs': 0.0
        }
        
        for i in range(num_queries):
            # Rastgele görüntü
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            response = self.query_with_boto3(image)
            
            if response['status'] == 'success':
                results['successful_queries'] += 1
                results['predictions'].append(response['predictions'])
            else:
                results['failed_queries'] += 1
            
            # AWS Rekognition cost: $0.001 per image
            results['costs'] += 0.001
            
            time.sleep(0.2)  # Rate limiting
            
            if (i + 1) % 25 == 0:
                print(f"Progress: {i+1}/{num_queries}")
        
        success_rate = results['successful_queries'] / num_queries
        print(f"✅ AWS attack completed! Success rate: {success_rate:.2%}")
        print(f"💰 Total cost: ${results['costs']:.2f}")
        
        return results


class CustomAPIAttack:
    """Generic custom API attack"""
    
    def __init__(self, endpoint: str, headers: Dict[str, str] = None, 
                 auth_token: str = None):
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
    
    def format_image_request(self, image: np.ndarray, format_type: str = "base64") -> Dict:
        """Görüntüyü API formatına çevir"""
        if format_type == "base64":
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            b64_image = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "image": b64_image,
                "format": "jpeg"
            }
        
        elif format_type == "array":
            return {
                "image": image.tolist(),
                "shape": list(image.shape)
            }
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def query_api(self, image: np.ndarray, format_type: str = "base64") -> Dict:
        """Custom API query"""
        payload = self.format_image_request(image, format_type)
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'response': response.json()
                }
            else:
                return {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def adaptive_attack(self, num_queries: int = 100) -> Dict:
        """Adaptive saldırı - API formatını otomatik keşfet"""
        print(f"🎯 Starting adaptive attack on {self.endpoint}")
        
        # Önce formatları test et
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        formats_to_try = ["base64", "array"]
        working_format = None
        
        for fmt in formats_to_try:
            print(f"Testing format: {fmt}")
            response = self.query_api(test_image, fmt)
            
            if response['status'] == 'success':
                working_format = fmt
                print(f"✅ Working format found: {fmt}")
                break
            else:
                print(f"❌ Format {fmt} failed: {response['error']}")
        
        if not working_format:
            print("❌ No working format found!")
            return {}
        
        # Ana saldırı
        results = {
            'format_used': working_format,
            'successful_queries': 0,
            'failed_queries': 0,
            'responses': []
        }
        
        for i in range(num_queries):
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            response = self.query_api(image, working_format)
            
            if response['status'] == 'success':
                results['successful_queries'] += 1
                results['responses'].append(response['response'])
            else:
                results['failed_queries'] += 1
            
            time.sleep(0.1)  # Conservative rate limiting
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{num_queries}")
        
        success_rate = results['successful_queries'] / num_queries
        print(f"✅ Adaptive attack completed! Success rate: {success_rate:.2%}")
        
        return results


class OpenSourceAPIAttack:
    """Open source model API attack (Hugging Face, etc.)"""
    
    def __init__(self, model_endpoint: str, api_token: str = None):
        self.endpoint = model_endpoint
        self.headers = {"Content-Type": "application/json"}
        
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
    
    def query_huggingface_api(self, image: np.ndarray) -> Dict:
        """Hugging Face Inference API query"""
        # Convert to bytes
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        try:
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.headers.get('Authorization', '').replace('Bearer ', '')}"},
                data=image_bytes,
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'predictions': response.json()
                }
            else:
                return {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def batch_attack_hf(self, num_queries: int = 100) -> Dict:
        """Hugging Face batch attack"""
        print(f"🎯 Starting Hugging Face API attack: {self.endpoint}")
        
        results = {
            'successful_queries': 0,
            'failed_queries': 0,
            'predictions': []
        }
        
        for i in range(num_queries):
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            response = self.query_huggingface_api(image)
            
            if response['status'] == 'success':
                results['successful_queries'] += 1
                results['predictions'].append(response['predictions'])
            else:
                results['failed_queries'] += 1
                if i < 5:  # İlk birkaç hatayı göster
                    print(f"Query {i+1} failed: {response['error']}")
            
            time.sleep(0.1)  # Generous rate limiting for free tier
            
            if (i + 1) % 25 == 0:
                print(f"Progress: {i+1}/{num_queries}")
        
        success_rate = results['successful_queries'] / num_queries
        print(f"✅ HF attack completed! Success rate: {success_rate:.2%}")
        
        return results


def demo_attacks():
    """Demo farklı API saldırıları"""
    print("🚀 AI Model Extraction - API-Specific Attack Demos")
    print("=" * 60)
    
    # Google Vision (API key gerekli)
    if False:  # Set to True if you have API key
        print("\n1. 🎯 Google Cloud Vision API Attack")
        google_attacker = GoogleVisionAttack("your-google-api-key")
        google_results = google_attacker.batch_attack(50)
    
    # AWS Rekognition (credentials gerekli)
    if False:  # Set to True if you have credentials
        print("\n2. 🎯 AWS Rekognition Attack")
        aws_attacker = AWSRekognitionAttack(
            "your-access-key",
            "your-secret-key"
        )
        aws_results = aws_attacker.batch_attack(50)
    
    # Custom API (test endpoint)
    print("\n3. 🎯 Custom API Attack")
    custom_attacker = CustomAPIAttack(
        endpoint="https://httpbin.org/post",  # Test endpoint
        headers={"Content-Type": "application/json"}
    )
    custom_results = custom_attacker.adaptive_attack(10)
    
    # Hugging Face (free tier)
    if False:  # Set to True for actual HF attack
        print("\n4. 🎯 Hugging Face API Attack")
        hf_attacker = OpenSourceAPIAttack(
            "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
        )
        hf_results = hf_attacker.batch_attack_hf(20)
    
    print("\n🎉 Demo completed!")
    print("⚠️  Remember: Only attack APIs you have permission to test!")


if __name__ == "__main__":
    demo_attacks()
