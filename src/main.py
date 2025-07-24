"""
Main Execution Script for Model Extraction Simulator
AI Model Çalma Simülatörü - Ana Çalıştırma Scripti
"""

import sys
import os
import argparse
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
from typing import Dict, Tuple

# Proje modülleri
from src.victim_model.victim_model import VictimModel, VictimModelAPI
from src.attacker.extraction_strategies import (
    RandomQueryStrategy, ActiveLearningStrategy, AdversarialQueryStrategy,
    ModelExtractor, KnowledgeDistillationTrainer
)
from src.clone_model.clone_model import CloneModel
from src.utils.utils import (
    load_config, load_dataset, evaluate_model_performance,
    calculate_model_similarity, save_experiment_results,
    create_experiment_directory, ExperimentLogger,
    set_random_seeds, get_device, plot_training_history,
    plot_model_comparison, count_parameters
)


class ModelExtractionExperiment:
    """Ana deney sınıfı"""
    
    def __init__(self, config_path: str):
        """
        Deney başlatma
        
        Args:
            config_path: Konfigürasyon dosyası yolu
        """
        self.config = load_config(config_path)
        self.device = get_device()
        
        # Rastgelelik seedlerini ayarla
        set_random_seeds(42)
        
        # Deney dizini oluştur
        self.experiment_dir = create_experiment_directory(
            self.config['experiment']['name']
        )
        
        # Logger oluştur
        log_file = os.path.join(self.experiment_dir, "logs", "experiment.log")
        self.logger = ExperimentLogger(log_file)
        self.logger.log_config(self.config)
        
        # Bileşenleri başlat
        self.victim_model = None
        self.victim_api = None
        self.clone_model = None
        self.extractor = None
        
        self.logger.log(f"Device: {self.device}")
        self.logger.log(f"Experiment directory: {self.experiment_dir}")
    
    def setup_victim_model(self):
        """Kurban modeli hazırla"""
        self.logger.log("Setting up victim model...")
        
        victim_config = self.config['victim_model']
        
        # Kurban modeli oluştur
        self.victim_model = VictimModel(
            model_type=victim_config['type'],
            num_classes=victim_config['num_classes'],
            pretrained=victim_config['pretrained'],
            device=self.device
        )
        
        # Eğer checkpoint varsa yükle
        if victim_config.get('checkpoint_path'):
            self.victim_model.load_checkpoint(victim_config['checkpoint_path'])
            self.logger.log(f"Loaded victim model from {victim_config['checkpoint_path']}")
        
        # API wrapper oluştur
        self.victim_api = VictimModelAPI(self.victim_model)
        
        # Model bilgilerini logla
        param_count = count_parameters(self.victim_model.model)
        self.logger.log(f"Victim model parameters: {param_count:,}")
        
        self.logger.log("Victim model setup complete")
    
    def setup_attacker(self):
        """Saldırgan bileşenlerini hazırla"""
        self.logger.log("Setting up attacker...")
        
        attack_config = self.config['attack']
        
        # Sorgulama stratejisi seç
        strategy_type = attack_config['strategy']
        
        if strategy_type == "random":
            query_strategy = RandomQueryStrategy(
                data_distribution="uniform",
                input_shape=(3, 32, 32)  # CIFAR-10 için
            )
        elif strategy_type == "active_learning":
            query_strategy = ActiveLearningStrategy(
                initial_pool_size=2000,
                uncertainty_method="entropy"
            )
        elif strategy_type == "adversarial":
            query_strategy = AdversarialQueryStrategy(
                epsilon=0.1,
                attack_method="fgsm"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        # Model extractor oluştur
        self.extractor = ModelExtractor(
            query_strategy=query_strategy,
            victim_api=self.victim_api,
            query_budget=attack_config['query_budget']
        )
        
        self.logger.log(f"Attacker setup complete with {strategy_type} strategy")
        self.logger.log(f"Query budget: {attack_config['query_budget']}")
    
    def setup_clone_model(self):
        """Klon modeli hazırla"""
        self.logger.log("Setting up clone model...")
        
        clone_config = self.config['clone_model']
        
        self.clone_model = CloneModel(
            architecture=clone_config['architecture'],
            num_classes=self.config['victim_model']['num_classes'],
            device=self.device
        )
        
        # Model bilgilerini logla
        param_count = count_parameters(self.clone_model.model)
        self.logger.log(f"Clone model parameters: {param_count:,}")
        
        self.logger.log("Clone model setup complete")
    
    def extract_knowledge(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bilgi çıkarma süreci"""
        self.logger.log("Starting knowledge extraction...")
        
        attack_config = self.config['attack']
        
        # Bilgi çıkarma
        stolen_queries, stolen_responses = self.extractor.extract_knowledge(
            batch_size=attack_config['batch_size']
        )
        
        # İstatistikleri logla
        extraction_stats = self.extractor.get_extraction_statistics()
        self.logger.log_results(extraction_stats)
        
        self.logger.log(f"Knowledge extraction complete")
        self.logger.log(f"Collected {len(stolen_queries)} query-response pairs")
        
        return stolen_queries, stolen_responses
    
    def train_clone_model(self, stolen_queries: np.ndarray, 
                         stolen_responses: np.ndarray) -> Dict:
        """Klon modeli eğitimi"""
        self.logger.log("Training clone model...")
        
        clone_config = self.config['clone_model']
        attack_config = self.config['attack']
        
        # Eğitim
        training_results = self.clone_model.train_with_stolen_data(
            stolen_queries=stolen_queries,
            stolen_labels=stolen_responses,
            epochs=clone_config['training_epochs'],
            learning_rate=clone_config['learning_rate'],
            batch_size=clone_config['batch_size'],
            temperature=attack_config.get('temperature', 3.0)
        )
        
        self.logger.log_results(training_results)
        
        # Eğitim grafiklerini çiz
        plot_path = os.path.join(self.experiment_dir, "plots", "training_history.png")
        plot_training_history(training_results, save_path=plot_path)
        
        # Modeli kaydet
        model_path = os.path.join(self.experiment_dir, "models", "clone_model.pth")
        self.clone_model.save_model(model_path)
        
        self.logger.log("Clone model training complete")
        return training_results
    
    def evaluate_models(self) -> Dict:
        """Model değerlendirmesi"""
        self.logger.log("Evaluating models...")
        
        # Test verisi yükle
        _, test_loader, _ = load_dataset(
            dataset_name=self.config['dataset']['name'],
            data_dir=self.config['dataset']['data_dir'],
            image_size=self.config['dataset']['image_size'],
            normalize=self.config['dataset']['normalize']
        )
        
        # Test verilerini tensor'a çevir
        test_inputs = []
        test_targets = []
        
        for batch_data, batch_targets in test_loader:
            test_inputs.append(batch_data)
            test_targets.append(batch_targets)
            if len(test_inputs) >= 10:  # İlk 10 batch'i al (performans için)
                break
        
        test_inputs = torch.cat(test_inputs, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        
        # Kurban model performansı
        victim_results = evaluate_model_performance(
            self.victim_model.model, 
            [(test_inputs, test_targets)], 
            self.device
        )
        
        # Klon model performansı
        clone_results = evaluate_model_performance(
            self.clone_model.model, 
            [(test_inputs, test_targets)], 
            self.device
        )
        
        # Model benzerliği
        victim_outputs = self.victim_model.query(test_inputs)['probabilities']
        clone_outputs = self.clone_model.predict(test_inputs)['probabilities']
        
        similarity_metrics = calculate_model_similarity(victim_outputs, clone_outputs)
        
        # Sonuçları birleştir
        evaluation_results = {
            'victim_performance': victim_results,
            'clone_performance': clone_results,
            'similarity_metrics': similarity_metrics,
            'fidelity': similarity_metrics['fidelity'],
            'query_budget_used': self.extractor.query_count,
            'query_budget_total': self.extractor.query_budget,
            'query_efficiency': similarity_metrics['fidelity'] / self.extractor.query_count * 1000  # Per 1000 queries
        }
        
        self.logger.log_results(evaluation_results)
        
        # Karşılaştırma grafiği
        comparison_data = {
            'accuracy': victim_results['accuracy'],
            'fidelity': similarity_metrics['fidelity'],
            'query_efficiency': evaluation_results['query_efficiency']
        }
        
        clone_comparison_data = {
            'accuracy': clone_results['accuracy'],
            'fidelity': similarity_metrics['fidelity'],
            'query_efficiency': evaluation_results['query_efficiency']
        }
        
        plot_path = os.path.join(self.experiment_dir, "plots", "model_comparison.png")
        plot_model_comparison(comparison_data, clone_comparison_data, save_path=plot_path)
        
        self.logger.log("Model evaluation complete")
        return evaluation_results
    
    def run_experiment(self) -> Dict:
        """Tam deney çalıştırma"""
        self.logger.log("Starting model extraction experiment...")
        
        try:
            # 1. Kurban modeli hazırla
            self.setup_victim_model()
            
            # 2. Saldırgan hazırla
            self.setup_attacker()
            
            # 3. Klon modeli hazırla
            self.setup_clone_model()
            
            # 4. Bilgi çıkarma
            stolen_queries, stolen_responses = self.extract_knowledge()
            
            # 5. Klon model eğitimi
            training_results = self.train_clone_model(stolen_queries, stolen_responses)
            
            # 6. Değerlendirme
            evaluation_results = self.evaluate_models()
            
            # 7. Sonuçları birleştir
            final_results = {
                'experiment_config': self.config,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'extraction_statistics': self.extractor.get_extraction_statistics(),
                'experiment_directory': self.experiment_dir
            }
            
            # 8. Sonuçları kaydet
            results_path = save_experiment_results(
                final_results, 
                self.config['experiment']['name'],
                self.experiment_dir
            )
            
            self.logger.log(f"Experiment completed successfully!")
            self.logger.log(f"Results saved to: {results_path}")
            
            return final_results
            
        except Exception as e:
            self.logger.log(f"Experiment failed: {str(e)}", level="ERROR")
            raise


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="AI Model Extraction Simulator")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    print("🚀 AI Model Extraction Simulator")
    print("=" * 50)
    
    try:
        # Deney çalıştır
        experiment = ModelExtractionExperiment(args.config)
        results = experiment.run_experiment()
        
        # Özet bilgiler
        print("\n📊 Experiment Summary:")
        print(f"Victim Model Accuracy: {results['evaluation_results']['victim_performance']['accuracy']:.4f}")
        print(f"Clone Model Accuracy: {results['evaluation_results']['clone_performance']['accuracy']:.4f}")
        print(f"Model Fidelity: {results['evaluation_results']['fidelity']:.4f}")
        print(f"Queries Used: {results['extraction_statistics']['total_queries']:,}")
        print(f"Query Efficiency: {results['evaluation_results']['query_efficiency']:.4f}")
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 Results directory: {results['experiment_directory']}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
