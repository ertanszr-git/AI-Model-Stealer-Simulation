"""
Utility Functions
Yardımcı fonksiyonlar ve veri işleme
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import yaml
import os
import json
from datetime import datetime


def load_config(config_path: str) -> Dict:
    """YAML konfigürasyon dosyasını yükle"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config: Dict, config_path: str):
    """Konfigürasyonu YAML dosyasına kaydet"""
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)


def load_dataset(dataset_name: str, data_dir: str = "./data", 
                 image_size: int = 32, normalize: bool = True) -> Tuple:
    """
    Veri setini yükle ve hazırla
    
    Args:
        dataset_name: Veri seti adı (cifar10, cifar100, imagenet)
        data_dir: Veri dizini
        image_size: Görüntü boyutu
        normalize: Normalizasyon uygula
        
    Returns:
        Tuple: (train_loader, test_loader, num_classes)
    """
    
    # Transform tanımla
    transform_list = [transforms.Resize((image_size, image_size))]
    
    if normalize:
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            # CIFAR normalizasyonu
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
        else:
            # ImageNet normalizasyonu
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), 
                                   (0.229, 0.224, 0.225))
            ])
    else:
        transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    
    # Veri seti yükle
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
        
    elif dataset_name.lower() == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform
        )
        num_classes = 100
        
    else:
        raise ValueError(f"Desteklenmeyen veri seti: {dataset_name}")
    
    # DataLoader oluştur
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader, num_classes


def calculate_model_similarity(model1_outputs: np.ndarray, 
                             model2_outputs: np.ndarray) -> Dict:
    """
    İki model arasındaki benzerlik metriklerini hesapla
    
    Args:
        model1_outputs: İlk modelin çıktıları
        model2_outputs: İkinci modelin çıktıları
        
    Returns:
        Dict: Benzerlik metrikleri
    """
    
    # Fidelity (Uyum) - Aynı sınıf tahmin etme oranı
    predictions1 = np.argmax(model1_outputs, axis=1)
    predictions2 = np.argmax(model2_outputs, axis=1)
    fidelity = np.mean(predictions1 == predictions2)
    
    # KL Divergence
    kl_divs = []
    for i in range(len(model1_outputs)):
        p = model1_outputs[i] + 1e-8  # Numerical stability
        q = model2_outputs[i] + 1e-8
        kl_div = np.sum(p * np.log(p / q))
        kl_divs.append(kl_div)
    
    avg_kl_divergence = np.mean(kl_divs)
    
    # Cosine Similarity
    cos_similarities = []
    for i in range(len(model1_outputs)):
        cos_sim = np.dot(model1_outputs[i], model2_outputs[i]) / (
            np.linalg.norm(model1_outputs[i]) * np.linalg.norm(model2_outputs[i])
        )
        cos_similarities.append(cos_sim)
    
    avg_cosine_similarity = np.mean(cos_similarities)
    
    return {
        'fidelity': fidelity,
        'kl_divergence': avg_kl_divergence,
        'cosine_similarity': avg_cosine_similarity,
        'agreement_rate': fidelity  # Alias
    }


def evaluate_model_performance(model, test_loader, device: str = "cpu") -> Dict:
    """
    Model performansını değerlendir
    
    Args:
        model: PyTorch modeli
        test_loader: Test veri yükleyicisi
        device: Cihaz (cpu/cuda)
        
    Returns:
        Dict: Performans metrikleri
    """
    
    model.eval()
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Sınıf bazında accuracy
            for i in range(target.size(0)):
                label = target[i].item()
                class_correct[label] = class_correct.get(label, 0) + (predicted[i] == target[i]).item()
                class_total[label] = class_total.get(label, 0) + 1
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Genel accuracy
    accuracy = correct / total
    
    # Sınıf bazında accuracy
    class_accuracies = {}
    for class_id in class_total:
        class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
    
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'targets': all_targets
    }


def plot_training_history(training_history: Dict, save_path: Optional[str] = None):
    """Eğitim geçmişini görselleştir"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss grafiği
    if 'losses' in training_history:
        axes[0].plot(training_history['losses'])
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
    
    # Accuracy grafiği
    if 'accuracies' in training_history:
        axes[1].plot(training_history['accuracies'])
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: List, y_pred: List, class_names: Optional[List] = None,
                         save_path: Optional[str] = None):
    """Confusion matrix görselleştir"""
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(victim_results: Dict, clone_results: Dict, 
                         save_path: Optional[str] = None):
    """Model karşılaştırması görselleştir"""
    
    metrics = ['accuracy', 'fidelity', 'query_efficiency']
    victim_values = [victim_results.get(m, 0) for m in metrics]
    clone_values = [clone_results.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, victim_values, width, label='Victim Model', alpha=0.8)
    ax.bar(x + width/2, clone_values, width, label='Clone Model', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_experiment_results(results: Dict, experiment_name: str, 
                           output_dir: str = "./experiments"):
    """Deney sonuçlarını kaydet"""
    
    # Output dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp ekle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # JSON serileştirme için numpy array'leri dönüştür
    serializable_results = convert_numpy_to_list(results)
    
    # Kaydet
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Sonuçlar kaydedildi: {filepath}")
    return filepath


def convert_numpy_to_list(obj):
    """Numpy array'leri liste'ye dönüştür (JSON serileştirme için)"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def load_experiment_results(filepath: str) -> Dict:
    """Deney sonuçlarını yükle"""
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def create_experiment_directory(experiment_name: str, base_dir: str = "./experiments") -> str:
    """Deney dizini oluştur"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Alt dizinler oluştur
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    return exp_dir


class ExperimentLogger:
    """Deney logları için yardımcı sınıf"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log(self, message: str, level: str = "INFO"):
        """Log mesajı yaz"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"{level}: {message}")
    
    def log_config(self, config: Dict):
        """Konfigürasyonu logla"""
        self.log("Experiment Configuration:")
        for key, value in config.items():
            self.log(f"  {key}: {value}")
    
    def log_results(self, results: Dict):
        """Sonuçları logla"""
        self.log("Experiment Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")


def set_random_seeds(seed: int = 42):
    """Rastgelelik seedlerini ayarla (tekrarlanabilirlik için)"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """En uygun cihazı seç"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


def count_parameters(model) -> int:
    """Model parametrelerini say"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def memory_usage() -> Dict:
    """Bellek kullanımını kontrol et"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    else:
        return {'message': 'CUDA not available'}
