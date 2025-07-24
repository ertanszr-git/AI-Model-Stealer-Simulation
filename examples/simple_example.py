"""
Simple Example: Basic Model Extraction
Basit örnek: Temel model çalma
"""

import sys
import os
from pathlib import Path

# Proje kök dizinini ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np

from src.victim_model.victim_model import VictimModel, VictimModelAPI
from src.attacker.extraction_strategies import RandomQueryStrategy, ModelExtractor
from src.clone_model.clone_model import CloneModel
from src.utils.utils import get_device, set_random_seeds


def simple_extraction_example():
    """Basit model çalma örneği"""
    
    print("🎯 Simple Model Extraction Example")
    print("=" * 40)
    
    # Device ve seed ayarla
    device = get_device()
    set_random_seeds(42)
    print(f"Device: {device}")
    
    # 1. Kurban modeli oluştur
    print("\n1. Creating victim model...")
    victim_model = VictimModel(
        model_type="resnet18",
        num_classes=10,
        pretrained=True,
        device=device
    )
    
    victim_api = VictimModelAPI(victim_model)
    print(f"✅ Victim model created with {sum(p.numel() for p in victim_model.model.parameters()):,} parameters")
    
    # 2. Saldırgan hazırla
    print("\n2. Setting up attacker...")
    query_strategy = RandomQueryStrategy(
        data_distribution="uniform",
        input_shape=(3, 32, 32)
    )
    
    extractor = ModelExtractor(
        query_strategy=query_strategy,
        victim_api=victim_api,
        query_budget=1000  # Küçük budget (demo için)
    )
    print("✅ Attacker setup complete")
    
    # 3. Bilgi çıkarma
    print("\n3. Extracting knowledge...")
    stolen_queries, stolen_responses = extractor.extract_knowledge(batch_size=32)
    
    extraction_stats = extractor.get_extraction_statistics()
    print(f"✅ Extracted {len(stolen_queries)} query-response pairs")
    print(f"📊 Budget used: {extraction_stats['total_queries']:,}/{extraction_stats['query_budget']:,}")
    
    # 4. Klon model oluştur ve eğit
    print("\n4. Training clone model...")
    clone_model = CloneModel(
        architecture="simple_cnn",  # Daha basit mimari
        num_classes=10,
        device=device
    )
    
    # Eğitim
    training_results = clone_model.train_with_stolen_data(
        stolen_queries=stolen_queries,
        stolen_labels=stolen_responses,
        epochs=20,  # Az epoch (demo için)
        learning_rate=0.01,
        batch_size=64,
        temperature=3.0
    )
    
    print(f"✅ Clone model trained")
    print(f"📈 Final training loss: {training_results['final_loss']:.4f}")
    print(f"📈 Final training accuracy: {training_results['final_accuracy']:.4f}")
    
    # 5. Basit değerlendirme
    print("\n5. Evaluating models...")
    
    # Test verisi oluştur
    test_queries = np.random.uniform(0, 1, (100, 3, 32, 32)).astype(np.float32)
    test_tensor = torch.from_numpy(test_queries)
    
    # Tahminleri al
    victim_predictions = victim_model.query(test_tensor)['predictions']
    clone_predictions = clone_model.predict(test_tensor)['predictions']
    
    # Fidelity hesapla
    fidelity = np.mean(victim_predictions == clone_predictions)
    
    print(f"📊 Model Fidelity: {fidelity:.4f}")
    print(f"📊 Query Efficiency: {fidelity / extraction_stats['total_queries'] * 1000:.4f} (per 1000 queries)")
    
    # 6. Sonuç
    print(f"\n✅ Simple extraction complete!")
    print(f"🎯 Successfully created a clone model with {fidelity:.1%} fidelity using {extraction_stats['total_queries']:,} queries")
    
    return {
        'fidelity': fidelity,
        'queries_used': extraction_stats['total_queries'],
        'training_loss': training_results['final_loss'],
        'training_accuracy': training_results['final_accuracy']
    }


if __name__ == "__main__":
    try:
        results = simple_extraction_example()
        print("\n🎉 Example completed successfully!")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
