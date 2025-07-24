# 🚀 Quick Start Guide - AI Model Extraction Simulator

Bu kılavuz size 5 dakikada nasıl başlayacağınızı gösterir.

## ⚡ Hızlı Başlangıç (5 dakika)

### 1. Kurulum (2 dakika)
```bash
# Projeyi klonla (zaten indirdiyseniz atla)
git clone <repository-url>
cd model-stealer

# Virtual environment ve dependencies
./setup.sh  # macOS/Linux
# veya
pip install -r requirements.txt
```

### 2. İlk Simülasyon (2 dakika)
```bash
# Güvenli eğitim moduna git
cd simulation/

# Basit simülasyon çalıştır
python simulator.py
```

### 3. Sonuçları İncele (1 dakika)
```bash
# Sonuçlar klasörüne bak
ls ../experiments/simulation_*/

# Fidelity skorunu gör
cat ../experiments/simulation_*/results/experiment_results.json | grep fidelity
```

**🎉 Tebrikler!** İlk model extraction simülasyonunuz tamamlandı.

---

## 🎯 Ne Gördünüz?

### Tipik Çıktı:
```
🔒 Safe simulation mode initialized
✅ Victim model ready: resnet18
🎯 Starting knowledge extraction (simulation)...
📊 Training clone model with stolen data...
🎉 Simulation completed successfully!

📊 Simulation Summary:
🎯 Fidelity: 94.52%
📊 Extraction Efficiency: 9.45
🔍 Queries Used: 10000
📁 Results: experiments/simulation_baseline_extraction_2024_07_24_15_30
```

### Bu Ne Anlama Geliyor?
- **Fidelity %94.52**: Klon model, orijinal modeli %94.52 doğrulukla taklit ediyor
- **10000 Sorgu**: Victim model'e 10,000 sorgu gönderdik
- **Extraction Efficiency 9.45**: Her 1000 sorgu için %9.45 fidelity kazandık

---

## 🎲 Farklı Deneyler Yapın

### Strateji Değiştir:
```bash
# Active learning ile dene (daha akıllı)
python simulator.py --config config/simulation.yaml
```

Config dosyasında değiştirin:
```yaml
attack:
  strategy: "active_learning"  # "random" yerine
  query_budget: 5000          # Daha az sorgu
```

### Model Değiştir:
```yaml
victim_model:
  type: "vgg16"     # "resnet18" yerine
  dataset: "cifar100"  # Daha zor dataset

clone_model:
  architecture: "lightweight_cnn"  # Daha küçük model
```

### Sonuçları Karşılaştır:
```bash
# Birkaç farklı deney çalıştır
python simulator.py  # Deney 1: ResNet18 vs ResNet18
# Config değiştir, sonra:
python simulator.py  # Deney 2: VGG16 vs LightweightCNN

# Sonuçları karşılaştır
ls ../experiments/simulation_*/results/
```

---

## 🔍 Sonuçları Derinlemesine İnceleyin

### Detaylı Analiz:
```bash
cd ../experiments/simulation_*/

# Model dosyalarını gör
ls models/
# clone_model.pth - Çalınan model

# Veri dosyalarını gör  
ls data/
# stolen_queries.npy - Çalınan sorgu verileri
# stolen_responses.npy - Victim model yanıtları

# Logları incele
cat logs/simulation.log
```

### Python'da Analiz:
```python
import json
import numpy as np

# Sonuçları yükle
with open('results/experiment_results.json') as f:
    results = json.load(f)

# Ana metrikleri yazdır
eval_results = results['evaluation_results']
print(f"Fidelity: {eval_results['fidelity']:.2%}")
print(f"Queries: {eval_results['query_budget_used']}")
print(f"Efficiency: {eval_results['extraction_efficiency']:.2f}")

# Çalınan verileri yükle
queries = np.load('data/stolen_queries.npy')
responses = np.load('data/stolen_responses.npy')
print(f"Stolen data shape: {queries.shape}")
```

---

## 🏆 Sonraki Adımlar

### 📚 Daha Fazla Öğrenin:
1. **Farklı stratejiler** deneyin (random, active_learning, adversarial)
2. **Model mimarileri** değiştirin (ResNet, VGG, Custom)
3. **Datasetsler** değiştirin (CIFAR-10, CIFAR-100, ImageNet)

### 🔬 Araştırma Yapın:
1. **Savunma yöntemleri** geliştirin
2. **Query efficiency** optimize edin  
3. **Custom strategies** yazın

### ⚠️ Real-World'e Geçin (Dikkatli!):
1. **Önce yasal izin** alın
2. **API documentation** okuyun
3. **Small budget** ile test edin

---

## ❓ Sıkça Sorulan Sorular

### Q: Fidelity neden düşük çıkıyor?
**A:** Birkaç sebep olabilir:
- Query budget çok düşük → Artırın (10000+ yapın)
- Strateji yetersiz → Active learning deneyin
- Model kapasitesi küçük → Clone model'i büyütün

### Q: Çok yavaş çalışıyor?
**A:** Optimizasyon için:
- Batch size'ı artırın (32 → 64)
- CPU yerine GPU kullanın (device: "cuda")
- Dataset'i küçültün (CIFAR-10 tercih edin)

### Q: Memory error alıyorum?
**A:** Memory azaltın:
- Batch size'ı düşürün (64 → 16)
- Model size'ı küçültün
- CPU kullanın (device: "cpu")

### Q: Real-world'e geçmek istiyorum?
**A:** Dikkatli olun:
1. Önce ETHICAL_GUIDELINES.md okuyun
2. Sadece kendi API'nizi test edin
3. real_world/ klasörüne bakın
4. --analyze-only ile başlayın

---

## 🎯 Başarı Kriterleri

Bu quick start'ı tamamladıktan sonra şunları öğrenmiş olacaksınız:

- ✅ Model extraction nasıl çalışır
- ✅ Fidelity, query efficiency nedir  
- ✅ Farklı stratejilerin etkisi
- ✅ Sonuçları nasıl yorumlamak
- ✅ İleri seviye deneyler nasıl yapılır

**🚀 Artık AI Model Extraction uzmanısınız!**

---

## 📞 Yardım

Takıldığınız yerde:
1. **REAL_WORLD_GUIDE.md** - Detaylı rehber
2. **ETHICAL_GUIDELINES.md** - Yasal bilgiler  
3. **GitHub Issues** - Teknik destek
4. **simulation/README.md** - Simülasyon detayları
