"""
Clone Model Implementation
Klon model - çalınan verilerle eğitilen model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import numpy as np


class CloneModel:
    """
    Klon model sınıfı - çalınan verilerle oluşturulan model
    """
    
    def __init__(self, architecture: str = "resnet18", num_classes: int = 10,
                 device: str = "cpu"):
        self.architecture = architecture
        self.num_classes = num_classes
        self.device = device
        
        # Model oluştur
        self.model = self._create_model(architecture, num_classes)
        self.model.to(device)
        
        # Eğitim geçmişi
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'fidelity_scores': []
        }
        
    def _create_model(self, architecture: str, num_classes: int) -> nn.Module:
        """Model mimarisini oluştur"""
        if architecture == "resnet18":
            model = models.resnet18(weights=None)  # Clone model için pretrained kullanmıyoruz
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture == "resnet50":
            model = models.resnet50(weights=None)  # Clone model için pretrained kullanmıyoruz
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture == "simple_cnn":
            model = SimpleCNN(num_classes)
        elif architecture == "lightweight":
            model = LightweightCNN(num_classes)
        else:
            raise ValueError(f"Desteklenmeyen mimari: {architecture}")
            
        return model
    
    def train_with_stolen_data(self, stolen_queries: np.ndarray, 
                              stolen_labels: np.ndarray,
                              validation_data: Optional[Tuple] = None,
                              epochs: int = 50,
                              learning_rate: float = 0.001,
                              batch_size: int = 64,
                              temperature: float = 3.0) -> Dict:
        """
        Çalınan verilerle modeli eğit
        
        Args:
            stolen_queries: Sorgular (X)
            stolen_labels: Soft labels veya hard labels (y)
            validation_data: Validasyon verisi (X_val, y_val)
            epochs: Eğitim epoch sayısı
            learning_rate: Öğrenme oranı
            batch_size: Batch boyutu
            temperature: Knowledge distillation sıcaklığı
            
        Returns:
            Dict: Eğitim sonuçları
        """
        
        # Dataset oluştur
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(stolen_queries).float(),
            torch.from_numpy(stolen_labels).float()
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Validation loader
        val_loader = None
        if validation_data:
            val_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(validation_data[0]).float(),
                torch.from_numpy(validation_data[1]).long()
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        # Optimizer ve loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Eğitim döngüsü
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_queries, batch_labels in train_loader:
                batch_queries = batch_queries.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_queries)
                
                # Loss hesaplama (soft labels için KL divergence)
                if len(batch_labels.shape) > 1 and batch_labels.shape[1] > 1:
                    # Soft labels - Knowledge Distillation
                    loss = self._kl_divergence_loss(outputs, batch_labels, temperature)
                else:
                    # Hard labels - Cross Entropy
                    loss = F.cross_entropy(outputs, batch_labels.long())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Accuracy hesaplama
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += batch_labels.size(0)
                
                if len(batch_labels.shape) > 1:
                    # Soft labels için accuracy
                    _, true_labels = torch.max(batch_labels.data, 1)
                    correct_predictions += (predicted == true_labels).sum().item()
                else:
                    correct_predictions += (predicted == batch_labels).sum().item()
            
            # Epoch sonuçları
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions
            
            self.training_history['losses'].append(avg_loss)
            self.training_history['accuracies'].append(accuracy)
            
            # Validation
            if val_loader:
                val_accuracy = self._evaluate(val_loader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                      f"Train Acc: {accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                      f"Accuracy: {accuracy:.4f}")
        
        return {
            'training_losses': self.training_history['losses'],
            'training_accuracies': self.training_history['accuracies'],
            'final_loss': self.training_history['losses'][-1],
            'final_accuracy': self.training_history['accuracies'][-1]
        }
    
    def _kl_divergence_loss(self, student_outputs: torch.Tensor, 
                           teacher_probs: torch.Tensor, 
                           temperature: float) -> torch.Tensor:
        """Knowledge Distillation için KL Divergence loss"""
        # Student softmax with temperature
        student_probs = F.softmax(student_outputs / temperature, dim=1)
        
        # Teacher probabilities (already softmax)
        teacher_probs_normalized = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
        
        # KL Divergence
        kl_loss = F.kl_div(
            torch.log(student_probs + 1e-8),
            teacher_probs_normalized,
            reduction='batchmean'
        )
        
        return kl_loss * (temperature ** 2)  # Temperature scaling
    
    def _evaluate(self, data_loader) -> float:
        """Model değerlendirme"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        self.model.train()
        return correct / total
    
    def predict(self, inputs: torch.Tensor) -> Dict:
        """Tahmin yap"""
        self.model.eval()
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            return {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'logits': outputs.cpu().numpy()
            }
    
    def calculate_fidelity(self, victim_model, test_inputs: torch.Tensor) -> float:
        """
        Kurban model ile fidelity (uyum) hesaplama
        Fidelity = Aynı tahminleri yapma oranı
        """
        self.model.eval()
        
        clone_predictions = self.predict(test_inputs)['predictions']
        victim_predictions = victim_model.query(test_inputs)['predictions']
        
        agreement = np.mean(clone_predictions == victim_predictions)
        return agreement
    
    def save_model(self, path: str):
        """Modeli kaydet"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Modeli yükle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})


class SimpleCNN(nn.Module):
    """Basit CNN mimarisi"""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # İlk konvolüsyon bloğu
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # İkinci konvolüsyon bloğu
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Üçüncü konvolüsyon bloğu
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LightweightCNN(nn.Module):
    """Hafif CNN mimarisi - daha az parametre"""
    
    def __init__(self, num_classes: int = 10):
        super(LightweightCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
