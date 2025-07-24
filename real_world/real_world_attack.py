"""
Real-world Model Extraction Attack
Gerçek dünya model çalma saldırısı
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
from typing import Dict, Tuple
import logging

# Proje modülleri
from adapters.real_world_adapter import RealWorldAPIAdapter, setup_real_world_attack
from adapters.target_analyzer import analyze_target_api
from src.attacker.extraction_strategies import (
    RandomQueryStrategy, ActiveLearningStrategy, 
    ModelExtractor, KnowledgeDistillationTrainer
)
from src.clone_model.clone_model import CloneModel
from src.utils.utils import (
    load_config, save_experiment_results,
    create_experiment_directory, ExperimentLogger,
    set_random_seeds, get_device
)


class RealWorldExtractor:
    """Gerçek dünya model çıkarma saldırısı"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.device = get_device()
        
        # Güvenlik kontrolü
        self._safety_check()
        
        # Rastgelelik seedlerini ayarla
        set_random_seeds(42)
        
        # Deney dizini oluştur
        self.experiment_dir = create_experiment_directory(
            self.config['experiment']['name']
        )
        
        # Logger oluştur
        log_file = os.path.join(self.experiment_dir, "logs", "real_world_attack.log")
        self.logger = ExperimentLogger(log_file)
        self.logger.log_config(self.config)
        
        # Bileşenler
        self.target_api = None
        self.clone_model = None
        self.extractor = None
        
        self.logger.log(f"Real-world extractor initialized")
        self.logger.log(f"Target: {self.config['target_api']['endpoint']}")
    
    def _safety_check(self):
        """Güvenlik kontrolü"""
        # API endpoint kontrolü
        endpoint = self.config['target_api']['endpoint']
        if 'localhost' not in endpoint and 'example.com' in endpoint:
            print("⚠️  WARNING: You're using example.com endpoint!")
            print("   Make sure to replace with your actual target API")
            
        # Rate limit kontrolü
        delay = self.config['target_api']['rate_limit_delay']
        if delay < 0.1:
            print("⚠️  WARNING: Very aggressive rate limiting!")
            print("   Consider increasing rate_limit_delay for ethical usage")
        
        # Budget kontrolü
        budget = self.config['attack']['query_budget']
        cost_per_request = self.config['target_api'].get('cost_per_request', 0)
        estimated_cost = budget * cost_per_request
        
        if estimated_cost > 10:  # $10
            print(f"⚠️  WARNING: Estimated cost ${estimated_cost:.2f}")
            response = input("   Continue? (y/N): ")
            if response.lower() != 'y':
                sys.exit("Attack cancelled by user")
    
    def analyze_target(self) -> Dict:
        """Hedef API'yi analiz et"""
        self.logger.log("Analyzing target API...")
        
        api_config = self.config['target_api']
        
        try:
            analysis = analyze_target_api(
                api_endpoint=api_config['endpoint'],
                api_key=api_config.get('api_key')
            )
            
            # Analiz sonuçlarını kaydet
            analysis_file = os.path.join(self.experiment_dir, "target_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            self.logger.log(f"Target analysis saved to {analysis_file}")
            return analysis
            
        except Exception as e:
            self.logger.log(f"Target analysis failed: {str(e)}", level="ERROR")
            return {}
    
    def setup_target_api(self):
        """Hedef API'yi setup et"""
        self.logger.log("Setting up target API adapter...")
        
        api_config = self.config['target_api']
        
        self.target_api = setup_real_world_attack(api_config)
        
        # Test connection
        try:
            # Create a test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            response = self.target_api.predict(test_image)
            
            if response['status'] == 'success':
                self.logger.log("✅ Target API connection successful")
            else:
                self.logger.log(f"⚠️ Target API test returned: {response}", level="WARNING")
                
        except Exception as e:
            self.logger.log(f"❌ Target API connection failed: {str(e)}", level="ERROR")
            raise
    
    def setup_attack_strategy(self):
        """Saldırı stratejisini hazırla"""
        self.logger.log("Setting up attack strategy...")
        
        attack_config = self.config['attack']
        
        # Sorgulama stratejisi seç
        strategy_type = attack_config['strategy']
        
        if strategy_type == "random":
            query_strategy = RandomQueryStrategy(
                data_distribution="uniform",
                input_shape=(3, self.config['dataset']['image_size'], 
                           self.config['dataset']['image_size'])
            )
        elif strategy_type == "active_learning":
            query_strategy = ActiveLearningStrategy(
                initial_pool_size=1000,
                uncertainty_method="entropy"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        # Model extractor oluştur
        self.extractor = ModelExtractor(
            query_strategy=query_strategy,
            victim_api=self.target_api,
            query_budget=attack_config['query_budget']
        )
        
        self.logger.log(f"Attack strategy ready: {strategy_type}")
    
    def setup_clone_model(self):
        """Klon modeli hazırla"""
        self.logger.log("Setting up clone model...")
        
        clone_config = self.config['clone_model']
        
        # Hedef API'den sınıf sayısını tahmin et
        estimated_classes = 1000  # ImageNet default, analiz sonucuna göre ayarlanabilir
        
        self.clone_model = CloneModel(
            architecture=clone_config['architecture'],
            num_classes=estimated_classes,
            device=self.device
        )
        
        self.logger.log("Clone model ready")
    
    def extract_knowledge(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bilgi çıkarma süreci"""
        self.logger.log("Starting knowledge extraction from real API...")
        
        attack_config = self.config['attack']
        
        # Stealth ayarları
        if attack_config.get('randomize_delays', False):
            self.extractor.randomize_delays = True
            self.extractor.min_delay = attack_config.get('min_delay', 0.5)
            self.extractor.max_delay = attack_config.get('max_delay', 2.0)
        
        # Bilgi çıkarma
        stolen_queries, stolen_responses = self.extractor.extract_knowledge(
            batch_size=attack_config['batch_size']
        )
        
        # Sonuçları kaydet
        if self.config['experiment'].get('save_queries', True):
            np.save(os.path.join(self.experiment_dir, "stolen_queries.npy"), stolen_queries)
        
        if self.config['experiment'].get('save_responses', True):
            np.save(os.path.join(self.experiment_dir, "stolen_responses.npy"), stolen_responses)
        
        # İstatistikleri logla
        extraction_stats = self.extractor.get_extraction_statistics()
        api_stats = self.target_api.get_model_info()
        
        self.logger.log_results(extraction_stats)
        self.logger.log_results(api_stats)
        
        # Maliyet hesaplama
        cost_per_request = self.config['target_api'].get('cost_per_request', 0)
        total_cost = extraction_stats['total_queries'] * cost_per_request
        self.logger.log(f"💰 Estimated cost: ${total_cost:.2f}")
        
        return stolen_queries, stolen_responses
    
    def train_clone_model(self, stolen_queries: np.ndarray, 
                         stolen_responses: np.ndarray) -> Dict:
        """Klon modeli eğitimi"""
        self.logger.log("Training clone model with stolen data...")
        
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
        
        # Modeli kaydet
        model_path = os.path.join(self.experiment_dir, "models", "stolen_clone_model.pth")
        self.clone_model.save_model(model_path)
        
        return training_results
    
    def evaluate_attack(self) -> Dict:
        """Saldırı değerlendirmesi"""
        self.logger.log("Evaluating attack success...")
        
        # Test verileri oluştur
        test_size = 100
        test_queries = np.random.uniform(0, 1, (test_size, 3, 224, 224)).astype(np.float32)
        
        # Her iki modelden tahmin al
        target_predictions = []
        clone_predictions = []
        
        self.logger.log("Getting predictions from target API...")
        for i, query in enumerate(test_queries):
            if i % 20 == 0:
                self.logger.log(f"Progress: {i}/{test_size}")
            
            target_resp = self.target_api.predict(query)
            target_predictions.append(target_resp['predictions'][0])
            
            clone_pred = self.clone_model.predict(torch.from_numpy(query).unsqueeze(0))
            clone_predictions.append(clone_pred['predictions'][0])
        
        # Fidelity hesapla
        target_predictions = np.array(target_predictions)
        clone_predictions = np.array(clone_predictions)
        
        fidelity = np.mean(target_predictions == clone_predictions)
        
        # Sonuçlar
        evaluation_results = {
            'fidelity': fidelity,
            'test_samples': test_size,
            'query_budget_used': self.extractor.query_count,
            'api_success_rate': self.target_api.get_model_info()['success_rate'],
            'total_cost': self.extractor.query_count * self.config['target_api'].get('cost_per_request', 0)
        }
        
        self.logger.log_results(evaluation_results)
        return evaluation_results
    
    def run_attack(self) -> Dict:
        """Tam saldırı çalıştırma"""
        self.logger.log("🚀 Starting real-world model extraction attack...")
        
        try:
            # 1. Hedef analizi
            target_analysis = self.analyze_target()
            
            # 2. API setup
            self.setup_target_api()
            
            # 3. Saldırı stratejisi
            self.setup_attack_strategy()
            
            # 4. Klon model
            self.setup_clone_model()
            
            # 5. Bilgi çıkarma
            stolen_queries, stolen_responses = self.extract_knowledge()
            
            # 6. Model eğitimi
            training_results = self.train_clone_model(stolen_queries, stolen_responses)
            
            # 7. Değerlendirme
            evaluation_results = self.evaluate_attack()
            
            # 8. Sonuçları birleştir
            final_results = {
                'target_analysis': target_analysis,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'extraction_statistics': self.extractor.get_extraction_statistics(),
                'api_statistics': self.target_api.get_model_info(),
                'experiment_directory': self.experiment_dir,
                'config': self.config
            }
            
            # 9. Sonuçları kaydet
            results_path = save_experiment_results(
                final_results,
                self.config['experiment']['name'],
                self.experiment_dir
            )
            
            self.logger.log(f"🎉 Real-world attack completed successfully!")
            self.logger.log(f"📁 Results saved to: {results_path}")
            
            return final_results
            
        except Exception as e:
            self.logger.log(f"💥 Attack failed: {str(e)}", level="ERROR")
            raise


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Real-world AI Model Extraction Attack")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/real_world.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze target API, don't run attack"
    )
    
    args = parser.parse_args()
    
    print("🎯 Real-world AI Model Extraction Attack")
    print("=" * 50)
    print("⚠️  ETHICAL WARNING:")
    print("   This tool is for educational and security research only!")
    print("   Ensure you have permission to test the target API.")
    print("   Be respectful of rate limits and terms of service.")
    print()
    
    # Kullanıcı onayı
    response = input("Do you have permission to test the target API? (y/N): ")
    if response.lower() != 'y':
        print("❌ Attack cancelled. Please ensure you have proper authorization.")
        sys.exit(1)
    
    try:
        extractor = RealWorldExtractor(args.config)
        
        if args.analyze_only:
            # Sadece analiz yap
            analysis = extractor.analyze_target()
            print("\n📊 Target Analysis Complete!")
            print(f"📁 Results in: {extractor.experiment_dir}")
        else:
            # Tam saldırı
            results = extractor.run_attack()
            
            # Özet
            print("\n📊 Attack Summary:")
            print(f"🎯 Fidelity: {results['evaluation_results']['fidelity']:.2%}")
            print(f"💰 Total Cost: ${results['evaluation_results']['total_cost']:.2f}")
            print(f"📊 Success Rate: {results['api_statistics']['success_rate']:.2%}")
            print(f"📁 Results: {results['experiment_directory']}")
        
    except Exception as e:
        print(f"\n❌ Attack failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
