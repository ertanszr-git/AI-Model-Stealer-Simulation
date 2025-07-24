# 🎯 Real-World Model Extraction Guide

Bu rehber, AI Model Extraction Simulator'ü gerçek dünya API'lerine karşı nasıl kullanacağınızı anlatır.

## ⚠️ Ethical Warning

**Bu araç sadece eğitim ve güvenlik araştırması içindir!**
- Hedef API'yi test etmek için izniniz olduğundan emin olun
- Rate limit'lere ve hizmet şartlarına saygı gösterin
- Sorumlu açıklama (responsible disclosure) prensiplerine uyun

## 🚀 Quick Start

### 1. Hedef API Seçimi

İlk olarak test etmek istediğiniz API'yi belirleyin:

**Popüler Seçenekler:**
- **Google Cloud Vision API** - Görüntü sınıflandırma
- **AWS Rekognition** - Nesne tanıma
- **Azure Computer Vision** - Görsel analiz
- **Custom ML APIs** - Kendi modeliniz
- **Open Source API'ler** - Test amaçlı

### 2. Konfigürasyon

`config/real_world.yaml` dosyasını hedef API'nize göre düzenleyin:

```yaml
target_api:
  endpoint: "https://your-target-api.com/predict"
  api_key: "your-api-key-here"
  headers:
    "Content-Type": "application/json"
    "Authorization": "Bearer your-token"
  
attack:
  query_budget: 1000      # Kaç sorgu yapacağınız
  strategy: "random"      # veya "active_learning"
  batch_size: 10         # Batch boyutu
```

### 3. Hızlı Test

Sadece hedef analizi yapmak için:

```bash
python real_world_attack.py --analyze-only
```

### 4. Tam Saldırı

```bash
python real_world_attack.py --config config/real_world.yaml
```

## 📋 Detaylı Adımlar

### Adım 1: Hedef Keşif

**Target Analyzer** kullanarak API'yi keşfedin:

```python
from src.utils.target_analyzer import analyze_target_api

# API analizi
analysis = analyze_target_api(
    api_endpoint="https://api.example.com/predict",
    api_key="your-key"
)

print(f"Rate limits: {analysis['rate_limits']}")
print(f"Input formats: {analysis['supported_formats']}")
print(f"Response structure: {analysis['response_structure']}")
```

### Adım 2: API Adapter Kurulumu

**Real-World Adapter** ile API'yi bağlayın:

```python
from src.utils.real_world_adapter import setup_real_world_attack

api_config = {
    'endpoint': 'https://api.example.com/predict',
    'api_key': 'your-key',
    'format': 'json',
    'rate_limit_delay': 1.0
}

target_api = setup_real_world_attack(api_config)

# Test bağlantısı
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
response = target_api.predict(test_image)
```

### Adım 3: Saldırı Stratejisi

**Mevcut Stratejiler:**

1. **Random Query**: Rastgele girdilerle hızlı tarama
2. **Active Learning**: Belirsizlik tabanlı akıllı sorgulama
3. **Adversarial**: Sınır durumları keşfetme

```python
from src.attacker.extraction_strategies import ModelExtractor, RandomQueryStrategy

strategy = RandomQueryStrategy(
    data_distribution="uniform",
    input_shape=(3, 224, 224)
)

extractor = ModelExtractor(
    query_strategy=strategy,
    victim_api=target_api,
    query_budget=1000
)
```

### Adım 4: Bilgi Çıkarma

```python
# Bilgi çıkarma süreci
stolen_queries, stolen_responses = extractor.extract_knowledge(
    batch_size=10
)

print(f"Extracted {len(stolen_queries)} query-response pairs")
```

### Adım 5: Klon Model Eğitimi

```python
from src.clone_model.clone_model import CloneModel

clone_model = CloneModel(
    architecture="lightweight_cnn",
    num_classes=1000
)

training_results = clone_model.train_with_stolen_data(
    stolen_queries=stolen_queries,
    stolen_labels=stolen_responses,
    epochs=50
)
```

## 🎯 Özel API Örnekleri

### Google Cloud Vision API

```yaml
target_api:
  endpoint: "https://vision.googleapis.com/v1/images:annotate"
  api_key: "your-google-api-key"
  format: "google_vision"
  headers:
    "Content-Type": "application/json"
  cost_per_request: 0.0015
```

### AWS Rekognition

```yaml
target_api:
  endpoint: "https://rekognition.us-east-1.amazonaws.com/"
  format: "aws_rekognition"
  aws_access_key: "your-access-key"
  aws_secret_key: "your-secret-key"
  region: "us-east-1"
  cost_per_request: 0.001
```

### Custom REST API

```yaml
target_api:
  endpoint: "https://your-api.com/predict"
  format: "json"
  headers:
    "Authorization": "Bearer your-token"
    "Content-Type": "application/json"
  input_format: "base64"
  response_field: "predictions"
```

## 📊 Saldırı Optimizasyonu

### Performance Tuning

1. **Rate Limiting**: API limitlerini aşmayın
   ```yaml
   rate_limit_delay: 1.0    # Sorgular arası bekleme
   randomize_delays: true   # Rastgele gecikme
   min_delay: 0.5
   max_delay: 2.0
   ```

2. **Batch Processing**: Mümkünse batch kullanın
   ```yaml
   batch_size: 50           # API'nin desteklediği maksimum
   ```

3. **Smart Querying**: Active learning kullanın
   ```yaml
   strategy: "active_learning"
   uncertainty_threshold: 0.5
   ```

### Cost Optimization

1. **Budget Planning**:
   ```yaml
   query_budget: 1000       # Maksimum sorgu sayısı
   cost_per_request: 0.001  # API maliyeti
   max_total_cost: 10.0     # Maksimum $10
   ```

2. **Progressive Extraction**:
   ```python
   # Küçük başlayıp artırın
   budgets = [100, 500, 1000, 2000]
   for budget in budgets:
       results = extract_with_budget(budget)
       if results['fidelity'] > 0.9:
           break
   ```

## 🛡️ Stealth Techniques

### 1. Traffic Patterns
```yaml
stealth:
  randomize_delays: true
  min_delay: 0.5
  max_delay: 3.0
  user_agent_rotation: true
  proxy_rotation: false  # Dikkatli kullanın
```

### 2. Input Diversity
```python
# Çeşitli input tipleri kullanın
strategies = [
    NaturalImageStrategy(),   # Doğal görüntüler
    SyntheticImageStrategy(), # Sentetik görüntüler
    NoiseStrategy()          # Gürültülü görüntüler
]
```

### 3. Gradual Extraction
```python
# Aşamalı saldırı
phases = [
    {'budget': 100, 'strategy': 'random'},
    {'budget': 300, 'strategy': 'active_learning'},
    {'budget': 500, 'strategy': 'adversarial'}
]
```

## 📈 Başarı Metrikleri

### 1. Fidelity (Doğruluk)
```python
fidelity = np.mean(target_predictions == clone_predictions)
print(f"Model fidelity: {fidelity:.2%}")
```

### 2. Query Efficiency
```python
efficiency = fidelity / (query_count / 1000)
print(f"Queries per 1% fidelity: {1/efficiency:.0f}")
```

### 3. Cost Effectiveness
```python
cost_per_fidelity = total_cost / fidelity
print(f"Cost per 1% fidelity: ${cost_per_fidelity:.2f}")
```

## 🚨 Güvenlik ve Etik

### Legal Compliance
- ✅ Test izni alın
- ✅ ToS'u okuyun
- ✅ Rate limit'lere uyun
- ✅ Veri gizliliğini koruyun

### Responsible Disclosure
```markdown
1. Keşfedilen güvenlik açıklarını rapor edin
2. API sağlayıcısına makul süre tanıyın
3. Açıklamadan önce izin alın
4. Detayları sorumlu şekilde paylaşın
```

### Detection Avoidance
```python
# Algılanmayı önleme teknikleri
detection_avoidance = {
    'rate_limiting': True,
    'request_randomization': True,
    'user_agent_rotation': True,
    'temporal_spreading': True
}
```

## 📁 Çıktı Dosyaları

Başarılı bir saldırı sonunda şu dosyalar oluşur:

```
experiments/real_world_attack_2024_01_15_10_30/
├── logs/
│   └── real_world_attack.log        # Detaylı loglar
├── models/
│   └── stolen_clone_model.pth       # Çalınan model
├── data/
│   ├── stolen_queries.npy           # Sorgu verileri
│   └── stolen_responses.npy         # API yanıtları
├── results/
│   └── experiment_results.json      # Sonuç metrikleri
└── target_analysis.json             # Hedef analizi
```

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Failed**
   ```python
   # Debug connection
   test_simple_request()
   check_api_key_validity()
   verify_endpoint_status()
   ```

2. **Rate Limit Exceeded**
   ```yaml
   # Config'de artırın
   rate_limit_delay: 2.0
   batch_size: 1
   ```

3. **Low Fidelity**
   ```python
   # Daha fazla query veya better strategy
   query_budget: 5000
   strategy: "active_learning"
   ```

4. **High Costs**
   ```yaml
   # Budget kontrolü
   query_budget: 500
   cost_per_request: 0.001
   max_total_cost: 5.0
   ```

## 📚 Advanced Usage

### Custom API Adapters
```python
class CustomAPIAdapter(RealWorldAPIAdapter):
    def _format_request(self, image):
        # Özel format logic
        return custom_format(image)
    
    def _parse_response(self, response):
        # Özel parsing logic
        return custom_parse(response)
```

### Multi-Target Attacks
```python
targets = [
    'google_vision_api',
    'aws_rekognition',
    'azure_vision'
]

for target in targets:
    run_extraction_attack(target)
```

### Ensemble Extraction
```python
# Birden fazla stratejinin birleşimi
ensemble_strategy = EnsembleStrategy([
    RandomQueryStrategy(),
    ActiveLearningStrategy(),
    AdversarialStrategy()
])
```

---

**⚠️ Son Uyarı**: Bu araçları kullanmadan önce mutlaka yasal izin alın ve etik kurallara uyun!
