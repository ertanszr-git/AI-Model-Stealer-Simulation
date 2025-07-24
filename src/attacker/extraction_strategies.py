"""
Attacker Implementation
Saldırgan bileşenleri - model çalma stratejileri
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod


class QueryStrategy(ABC):
    """Sorgulama stratejisi için abstract base class"""
    
    @abstractmethod
    def select_queries(self, budget: int, **kwargs) -> np.ndarray:
        """Sorgulanacak verileri seç"""
        pass


class RandomQueryStrategy(QueryStrategy):
    """Rastgele sorgulama stratejisi"""
    
    def __init__(self, data_distribution: str = "uniform", input_shape: Tuple = (3, 32, 32)):
        self.data_distribution = data_distribution
        self.input_shape = input_shape
        
    def select_queries(self, budget: int, **kwargs) -> np.ndarray:
        """Rastgele girdi örnekleri oluştur"""
        if self.data_distribution == "uniform":
            # [0, 1] aralığında uniform dağılım
            queries = np.random.uniform(0, 1, (budget, *self.input_shape))
        elif self.data_distribution == "normal":
            # Normal dağılım (ImageNet normalize edilmiş)
            queries = np.random.normal(0, 1, (budget, *self.input_shape))
        else:
            # [0, 255] aralığında uniform (doğal görüntü benzeri)
            queries = np.random.uniform(0, 255, (budget, *self.input_shape)) / 255.0
            
        return queries.astype(np.float32)


class ActiveLearningStrategy(QueryStrategy):
    """Aktif öğrenme tabanlı sorgulama stratejisi"""
    
    def __init__(self, initial_pool_size: int = 1000, uncertainty_method: str = "entropy"):
        self.initial_pool_size = initial_pool_size
        self.uncertainty_method = uncertainty_method
        self.candidate_pool = None
        
    def select_queries(self, budget: int, victim_api=None, clone_model=None, **kwargs) -> np.ndarray:
        """Belirsizlik tabanlı örnek seçimi"""
        if self.candidate_pool is None:
            # İlk havuzu rastgele oluştur
            self.candidate_pool = np.random.uniform(0, 1, (self.initial_pool_size, 3, 32, 32))
        
        if clone_model is None:
            # Clone model yoksa rastgele seç
            indices = np.random.choice(len(self.candidate_pool), budget, replace=False)
            return self.candidate_pool[indices]
        
        # Clone model ile belirsizlik hesapla
        uncertainties = self._calculate_uncertainty(self.candidate_pool, clone_model)
        
        # En belirsiz örnekleri seç
        top_indices = np.argsort(uncertainties)[-budget:]
        selected_queries = self.candidate_pool[top_indices]
        
        # Seçilen örnekleri havuzdan çıkar
        remaining_indices = np.setdiff1d(np.arange(len(self.candidate_pool)), top_indices)
        self.candidate_pool = self.candidate_pool[remaining_indices]
        
        return selected_queries
    
    def _calculate_uncertainty(self, data: np.ndarray, model) -> np.ndarray:
        """Belirsizlik hesapla"""
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for i in range(0, len(data), 32):  # Batch processing
                batch = torch.from_numpy(data[i:i+32]).float()
                if torch.cuda.is_available():
                    batch = batch.cuda()
                    
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                
                if self.uncertainty_method == "entropy":
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    uncertainties.extend(entropy.cpu().numpy())
                elif self.uncertainty_method == "margin":
                    # En yüksek iki olasılık arasındaki fark
                    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                    uncertainties.extend((-margin).cpu().numpy())  # Negatif çünkü küçük margin = yüksek belirsizlik
                    
        return np.array(uncertainties)


class AdversarialQueryStrategy(QueryStrategy):
    """Adversarial sorgulama stratejisi"""
    
    def __init__(self, epsilon: float = 0.1, attack_method: str = "fgsm"):
        self.epsilon = epsilon
        self.attack_method = attack_method
        
    def select_queries(self, budget: int, base_images: np.ndarray = None, 
                      victim_api=None, **kwargs) -> np.ndarray:
        """Adversarial örnekler oluştur"""
        if base_images is None:
            # Rastgele base görüntüler oluştur
            base_images = np.random.uniform(0, 1, (budget, 3, 32, 32))
            
        if victim_api is None:
            return base_images
            
        adversarial_queries = []
        
        for i in range(min(budget, len(base_images))):
            img = base_images[i:i+1]
            
            if self.attack_method == "fgsm":
                adv_img = self._fgsm_attack(img, victim_api)
            else:
                adv_img = img  # Fallback
                
            adversarial_queries.append(adv_img)
            
        return np.concatenate(adversarial_queries, axis=0)
    
    def _fgsm_attack(self, image: np.ndarray, victim_api) -> np.ndarray:
        """Fast Gradient Sign Method saldırısı"""
        # Basit FGSM implementasyonu
        # Gerçek implementasyon gradyan hesaplama gerektirir
        # Burada rastgele noise ekliyoruz (simülasyon için)
        noise = np.random.uniform(-self.epsilon, self.epsilon, image.shape)
        adversarial = np.clip(image + noise, 0, 1)
        return adversarial


class ModelExtractor:
    """Ana model çalma sınıfı"""
    
    def __init__(self, query_strategy: QueryStrategy, victim_api, 
                 query_budget: int = 10000):
        self.query_strategy = query_strategy
        self.victim_api = victim_api
        self.query_budget = query_budget
        self.stolen_data = []
        self.query_count = 0
        
    def extract_knowledge(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kurban modelden bilgi çıkarma
        
        Returns:
            Tuple[queries, responses]: Sorgular ve yanıtlar
        """
        all_queries = []
        all_responses = []
        
        remaining_budget = self.query_budget
        
        while remaining_budget > 0:
            current_batch_size = min(batch_size, remaining_budget)
            
            # Sorgular oluştur
            queries = self.query_strategy.select_queries(
                budget=current_batch_size,
                victim_api=self.victim_api
            )
            
            # Kurban modeli sorgula
            responses = self._query_victim_batch(queries)
            
            all_queries.append(queries)
            all_responses.append(responses)
            
            remaining_budget -= current_batch_size
            self.query_count += current_batch_size
            
            if self.query_count % 1000 == 0:
                print(f"Sorgu sayısı: {self.query_count}/{self.query_budget}")
        
        # Tüm verileri birleştir
        final_queries = np.concatenate(all_queries, axis=0)
        final_responses = np.concatenate(all_responses, axis=0)
        
        self.stolen_data = (final_queries, final_responses)
        return final_queries, final_responses
    
    def _query_victim_batch(self, queries: np.ndarray) -> np.ndarray:
        """Batch halinde kurban modeli sorgula"""
        responses = []
        
        for query in queries:
            # API formatında sorgu
            api_response = self.victim_api.predict(query, return_probabilities=True)
            
            if api_response.get('status') == 'success':
                # Soft labels (olasılıklar) kullan
                responses.append(api_response['probabilities'][0])
            else:
                # Hata durumunda rastgele yanıt
                num_classes = 10  # CIFAR-10 için
                responses.append(np.random.uniform(0, 1, num_classes))
                
        return np.array(responses)
    
    def get_extraction_statistics(self) -> Dict:
        """Çıkarma istatistikleri"""
        return {
            'total_queries': self.query_count,
            'query_budget': self.query_budget,
            'budget_used_percent': (self.query_count / self.query_budget) * 100,
            'data_collected': len(self.stolen_data[0]) if self.stolen_data else 0,
            'strategy_type': type(self.query_strategy).__name__
        }


class KnowledgeDistillationTrainer:
    """Çalınan verilerle model eğitimi"""
    
    def __init__(self, student_model: nn.Module, temperature: float = 3.0,
                 alpha: float = 0.7):
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha  # Knowledge distillation weight
        
    def train_with_stolen_data(self, stolen_queries: np.ndarray, 
                              stolen_responses: np.ndarray,
                              epochs: int = 50, lr: float = 0.001) -> Dict:
        """Çalınan verilerle öğrenci modeli eğit"""
        
        # Dataset oluştur
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(stolen_queries).float(),
            torch.from_numpy(stolen_responses).float()
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        
        # Eğitim döngüsü
        training_losses = []
        self.student_model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_queries, batch_responses in dataloader:
                if torch.cuda.is_available():
                    batch_queries = batch_queries.cuda()
                    batch_responses = batch_responses.cuda()
                
                optimizer.zero_grad()
                
                # Öğrenci model çıktısı
                student_outputs = self.student_model(batch_queries)
                
                # Knowledge distillation loss
                loss = self._distillation_loss(student_outputs, batch_responses)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'epochs_trained': epochs
        }
    
    def _distillation_loss(self, student_outputs: torch.Tensor, 
                          teacher_probs: torch.Tensor) -> torch.Tensor:
        """Knowledge distillation loss hesapla"""
        # Soft targets ile KL divergence
        student_probs = torch.softmax(student_outputs / self.temperature, dim=1)
        teacher_probs_temp = torch.softmax(teacher_probs * self.temperature, dim=1)
        
        kl_loss = torch.nn.functional.kl_div(
            torch.log(student_probs + 1e-8),
            teacher_probs_temp,
            reduction='batchmean'
        )
        
        return kl_loss
