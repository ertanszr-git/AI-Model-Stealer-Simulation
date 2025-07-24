"""
AI Model Extraction Simulator - Safe Educational Mode
Güvenli eğitim ve test ortamı için model çıkarma simülatörü
"""

import sys
import os
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
import argparse
from typing import Dict

# Proje modülleri
from src.victim_model.victim_model import VictimModel, VictimModelAPI
from src.attacker.extraction_strategies import (
    RandomQueryStrategy, ActiveLearningStrategy, 
    AdversarialQueryStrategy, ModelExtractor
)
from src.clone_model.clone_model import CloneModel
from src.utils.utils import (
    load_config, save_experiment_results,
    create_experiment_directory, ExperimentLogger,
    set_random_seeds, get_device
)


class SafeModelExtractor:
    """Güvenli simülasyon ortamı için model extractor"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.device = get_device()
        
        # Güvenlik kontrolü - sadece simülasyon modunda çalışır
        self._safety_check()
        
        # Rastgelelik seedlerini ayarla
        set_random_seeds(42)
        
        # Deney dizini oluştur
        self.experiment_dir = create_experiment_directory(
            f"simulation_{self.config['experiment']['name']}"
        )
        
        # Logger oluştur
        log_file = os.path.join(self.experiment_dir, "logs", "simulation.log")
        self.logger = ExperimentLogger(log_file)
        self.logger.log_config(self.config)
        
        # Bileşenler
        self.victim_model = None
        self.victim_api = None
        self.clone_model = None
        self.extractor = None
        
        self.logger.log("🔒 Safe simulation mode initialized")
    
    def _safety_check(self):
        """Güvenlik kontrolü - sadece simülasyon"""
        # Real-world config kontrolü
        if 'target_api' in self.config:
            raise ValueError("❌ Real-world config detected! Use simulation config only.")
        
        # Endpoint kontrolü
        victim_config = self.config.get('victim_model', {})
        if 'endpoint' in victim_config:
            raise ValueError("❌ External endpoint detected! Simulation mode only.")
        
        print("✅ Safety check passed - Simulation mode only")
    
    def setup_victim_model(self):
        """Güvenli victim model setup"""
        self.logger.log("Setting up victim model for simulation...")
        
        victim_config = self.config['victim_model']
        
        # Local model oluştur
        self.victim_model = VictimModel(
            model_type=victim_config['type'],
            num_classes=victim_config['num_classes'],
            pretrained=victim_config['pretrained'],
            device=self.device
        )
        
        # API wrapper
        self.victim_api = VictimModelAPI(self.victim_model)
        
        self.logger.log(f"✅ Victim model ready: {victim_config['type']}")
        self.logger.log(f"📊 Dataset: {victim_config['dataset']} ({victim_config['num_classes']} classes)")
    
    def setup_attack_strategy(self):
        """Saldırı stratejisini hazırla"""
        self.logger.log("Setting up attack strategy...")
        
        attack_config = self.config['attack']
        strategy_type = attack_config['strategy']
        
        # Dataset boyutlarını al
        image_size = self.config['dataset']['image_size']
        
        if strategy_type == "random":
            query_strategy = RandomQueryStrategy(
                data_distribution="uniform",
                input_shape=(3, image_size, image_size)
            )
        elif strategy_type == "active_learning":
            query_strategy = ActiveLearningStrategy(
                initial_pool_size=1000,
                uncertainty_method="entropy"
            )
        elif strategy_type == "adversarial":
            query_strategy = AdversarialQueryStrategy(
                epsilon=0.03,
                attack_type="fgsm"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        # Model extractor oluştur
        self.extractor = ModelExtractor(
            query_strategy=query_strategy,
            victim_api=self.victim_api,
            query_budget=attack_config['query_budget']
        )
        
        self.logger.log(f"✅ Attack strategy ready: {strategy_type}")
    
    def setup_clone_model(self):
        """Klon modeli hazırla"""
        self.logger.log("Setting up clone model...")
        
        clone_config = self.config['clone_model']
        victim_config = self.config['victim_model']
        
        self.clone_model = CloneModel(
            architecture=clone_config['architecture'],
            num_classes=victim_config['num_classes'],
            device=self.device
        )
        
        self.logger.log(f"✅ Clone model ready: {clone_config['architecture']}")
    
    def run_extraction(self) -> Dict:
        """Bilgi çıkarma süreci"""
        self.logger.log("🎯 Starting knowledge extraction (simulation)...")
        
        attack_config = self.config['attack']
        
        # Bilgi çıkarma
        stolen_queries, stolen_responses = self.extractor.extract_knowledge(
            batch_size=attack_config['batch_size']
        )
        
        # Sonuçları kaydet
        np.save(os.path.join(self.experiment_dir, "stolen_queries.npy"), stolen_queries)
        np.save(os.path.join(self.experiment_dir, "stolen_responses.npy"), stolen_responses)
        
        # İstatistikleri logla
        extraction_stats = self.extractor.get_extraction_statistics()
        victim_stats = self.victim_api.get_model_info()
        
        self.logger.log_results(extraction_stats)
        self.logger.log_results(victim_stats)
        
        return stolen_queries, stolen_responses
    
    def train_clone_model(self, stolen_queries: np.ndarray, 
                         stolen_responses: np.ndarray) -> Dict:
        """Klon modeli eğitimi"""
        self.logger.log("🔧 Training clone model with stolen data...")
        
        clone_config = self.config['clone_model']
        attack_config = self.config['attack']
        
        # Eğitim
        training_results = self.clone_model.train_with_stolen_data(
            stolen_queries=stolen_queries,
            stolen_labels=stolen_responses,
            epochs=clone_config['training_epochs'],
            learning_rate=clone_config['learning_rate'],
            batch_size=clone_config['batch_size'],
            temperature=attack_config.get('temperature', 1.0)
        )
        
        self.logger.log_results(training_results)
        
        # Modeli kaydet
        model_path = os.path.join(self.experiment_dir, "models", "clone_model.pth")
        self.clone_model.save_model(model_path)
        
        return training_results
    
    def evaluate_simulation(self) -> Dict:
        """Simülasyon değerlendirmesi"""
        self.logger.log("📊 Evaluating simulation results...")
        
        # Test verileri oluştur
        test_size = 1000
        image_size = self.config['dataset']['image_size']
        test_queries = np.random.uniform(0, 1, (test_size, 3, image_size, image_size)).astype(np.float32)
        
        # Predictions
        victim_predictions = []
        clone_predictions = []
        
        for query in test_queries:
            # Victim model
            victim_resp = self.victim_api.predict(query)
            victim_predictions.append(victim_resp['predictions'][0])  # İlk elemanı al
            
            # Clone model
            clone_pred = self.clone_model.predict(torch.from_numpy(query).unsqueeze(0))
            clone_predictions.append(clone_pred['predictions'][0])  # İlk elemanı al
        
        # Metrics
        victim_predictions = np.array(victim_predictions)
        clone_predictions = np.array(clone_predictions)
        
        fidelity = np.mean(victim_predictions == clone_predictions)
        
        evaluation_results = {
            'fidelity': fidelity,
            'test_samples': test_size,
            'query_budget_used': self.extractor.query_count,
            'extraction_efficiency': fidelity / (self.extractor.query_count / 1000)
        }
        
        self.logger.log_results(evaluation_results)
        return evaluation_results
    
    def run_simulation(self) -> Dict:
        """Tam simülasyon çalıştırma"""
        self.logger.log("🚀 Starting safe model extraction simulation...")
        
        try:
            # 1. Victim model setup
            self.setup_victim_model()
            
            # 2. Attack strategy
            self.setup_attack_strategy()
            
            # 3. Clone model
            self.setup_clone_model()
            
            # 4. Knowledge extraction
            stolen_queries, stolen_responses = self.run_extraction()
            
            # 5. Model training
            training_results = self.train_clone_model(stolen_queries, stolen_responses)
            
            # 6. Evaluation
            evaluation_results = self.evaluate_simulation()
            
            # 7. Final results
            final_results = {
                'simulation_type': 'safe_educational',
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'extraction_statistics': self.extractor.get_extraction_statistics(),
                'victim_statistics': self.victim_api.get_model_info(),
                'experiment_directory': self.experiment_dir,
                'config': self.config
            }
            
            # 8. Save results
            results_path = save_experiment_results(
                final_results,
                f"simulation_{self.config['experiment']['name']}",
                self.experiment_dir
            )
            
            self.logger.log(f"🎉 Simulation completed successfully!")
            self.logger.log(f"📁 Results saved to: {results_path}")
            
            return final_results
            
        except Exception as e:
            self.logger.log(f"💥 Simulation failed: {str(e)}", level="ERROR")
            raise


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="AI Model Extraction Simulator - Safe Mode")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/simulation.yaml",
        help="Simulation configuration file path"
    )
    
    args = parser.parse_args()
    
    print("🔒 AI Model Extraction Simulator - Safe Educational Mode")
    print("=" * 60)
    print("✅ This is a SAFE simulation environment")
    print("✅ No real APIs will be attacked")
    print("✅ Only local models are used")
    print("✅ Perfect for learning and experimentation")
    print()
    
    try:
        simulator = SafeModelExtractor(args.config)
        results = simulator.run_simulation()
        
        # Summary
        print("\n📊 Simulation Summary:")
        print(f"🎯 Fidelity: {results['evaluation_results']['fidelity']:.2%}")
        print(f"📊 Extraction Efficiency: {results['evaluation_results']['extraction_efficiency']:.2f}")
        print(f"🔍 Queries Used: {results['extraction_statistics']['total_queries']}")
        print(f"📁 Results: {results['experiment_directory']}")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
