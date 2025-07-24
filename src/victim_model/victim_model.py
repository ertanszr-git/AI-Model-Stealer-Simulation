"""
Victim Model Implementation
Kurban model - gerçek dünyada çalmaya çalışılan model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import numpy as np


class VictimModel:
    """
    Kurban model sınıfı - saldırganın erişmeye çalıştığı model
    """
    
    def __init__(self, model_type: str = "resnet18", num_classes: int = 10, 
                 pretrained: bool = True, device: str = "cpu"):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        self.query_count = 0
        self.query_log = []
        
        # Model oluştur
        self.model = self._create_model(model_type, num_classes, pretrained)
        self.model.to(device)
        self.model.eval()
        
    def _create_model(self, model_type: str, num_classes: int, pretrained: bool) -> nn.Module:
        """Model mimarisini oluştur"""
        if model_type == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vgg16(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
        
        return model
    
    def query(self, inputs: torch.Tensor, return_logits: bool = False, 
              temperature: float = 1.0) -> Dict:
        """
        Model sorgulama - saldırganın kullanacağı API
        
        Args:
            inputs: Giriş tensörü
            return_logits: Ham logit değerlerini döndür
            temperature: Softmax sıcaklığı
            
        Returns:
            Dict: Tahmin sonuçları
        """
        self.query_count += len(inputs)
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            logits = self.model(inputs)
            
            # Sıcaklık uygulaması
            scaled_logits = logits / temperature
            probabilities = F.softmax(scaled_logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Sorgu logunu kaydet
            self._log_query(inputs, logits, predictions)
            
            result = {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'confidence': torch.max(probabilities, dim=1)[0].cpu().numpy()
            }
            
            if return_logits:
                result['logits'] = logits.cpu().numpy()
                
            return result
    
    def _log_query(self, inputs: torch.Tensor, logits: torch.Tensor, 
                   predictions: torch.Tensor):
        """Sorgu geçmişini kaydet"""
        import time
        query_info = {
            'timestamp': time.time(),  # Use system time instead of CUDA
            'input_shape': inputs.shape,
            'predictions': predictions.cpu().numpy(),
            'max_logit': torch.max(logits, dim=1)[0].cpu().numpy(),
        }
        self.query_log.append(query_info)
    
    def get_query_statistics(self) -> Dict:
        """Sorgu istatistiklerini döndür"""
        return {
            'total_queries': self.query_count,
            'query_history_length': len(self.query_log),
            'average_confidence': np.mean([np.mean(q['max_logit']) for q in self.query_log]) if self.query_log else 0
        }
    
    def reset_query_log(self):
        """Sorgu logunu temizle"""
        self.query_count = 0
        self.query_log = []
    
    def load_checkpoint(self, checkpoint_path: str):
        """Model ağırlıklarını yükle"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
    def save_checkpoint(self, checkpoint_path: str):
        """Model ağırlıklarını kaydet"""
        torch.save(self.model.state_dict(), checkpoint_path)


class VictimModelAPI:
    """
    Kurban model için web API simülatörü
    Gerçek dünyada bu bir REST API olurdu
    """
    
    def __init__(self, victim_model: VictimModel, rate_limit: Optional[int] = None):
        self.victim_model = victim_model
        self.rate_limit = rate_limit
        self.request_count = 0
        
    def predict(self, image_data: np.ndarray, return_probabilities: bool = True) -> Dict:
        """
        API tahmin endpoint'i
        
        Args:
            image_data: Görüntü verisi (numpy array)
            return_probabilities: Olasılık değerlerini döndür
            
        Returns:
            Dict: API yanıtı
        """
        # Rate limiting kontrolü
        if self.rate_limit and self.request_count >= self.rate_limit:
            return {'error': 'Rate limit exceeded'}
        
        self.request_count += 1
        
        # Numpy array'i tensor'a çevir
        if isinstance(image_data, np.ndarray):
            inputs = torch.from_numpy(image_data).float()
            if len(inputs.shape) == 3:  # Tek görüntü
                inputs = inputs.unsqueeze(0)
        else:
            inputs = image_data
            
        # Model sorgulama
        result = self.victim_model.query(inputs)
        
        # API formatında yanıt
        api_response = {
            'status': 'success',
            'predictions': result['predictions'].tolist()
        }
        
        if return_probabilities:
            api_response['probabilities'] = result['probabilities'].tolist()
            
        return api_response
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür (gerçek API'lerde mevcut olabilir)"""
        return {
            'model_type': self.victim_model.model_type,
            'num_classes': self.victim_model.num_classes,
            'input_shape': [3, 32, 32],  # CIFAR-10 için
            'available_endpoints': ['predict', 'model_info']
        }
