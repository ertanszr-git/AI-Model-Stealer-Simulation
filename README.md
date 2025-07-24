# AI Model Extraction Simulator
## Makine Öğrenimi Model Çalma Simülatörü

**Eğitim ve güvenlik araştırması için** geliştirilmiş kapsamlı model çıkarma simülatörü.

---

## 🚀 Hızlı Başlangıç

**⚡ 5 dakikada başlamak istiyorsanız**: [`QUICK_START.md`](QUICK_START.md) dosyasına bakın!

### 🔒 Güvenli Mod (Önerilen):
```bash
cd simulation/
python simulator.py
```

### ⚠️ Real-World Mod (İzin Gerekli):
```bash
cd real_world/
python real_world_attack.py --analyze-only
```

---

## 📖 İçindekiler

- [🎯 Ne Yapar Bu Simulator?](#-ne-yapar-bu-simulator)
- [� Proje Yapısı](#-proje-yapısı)
- [🔧 Kurulum](#-kurulum)
- [�🔒 Güvenli Eğitim Modülü](#-mod-1-güvenli-eğitim-simülatörü-önerilen)
- [⚠️ Real-World Attack Modülü](#️-mod-2-real-world-api-saldırıları)
- [� Karşılaştırma Tablosu](#-karşılaştırma-tablosu)
- [🔧 Teknik Özellikler](#-teknik-özellikler)
- [🔧 Troubleshooting](#-troubleshooting)
- [📚 Eğitim Amaçları](#-eğitim-amaçları)

---

## 🎯 Ne Yapar Bu Simulator?

Bu proje, **AI model extraction** saldırılarını simüle eder:

1. **🎯 Target Model**: Hedef AI modelini taklit eder
2. **🕵️ Attack Strategies**: Farklı saldırı tekniklerini uygular  
3. **🤖 Clone Training**: Çalınan verilerle klon model eğitir
4. **📊 Success Evaluation**: Saldırının başarısını ölçer

### Desteklenen Teknikler:
- **Query Strategies**: Random, Active Learning, Adversarial
- **Model Architectures**: ResNet, VGG, Custom CNNs
- **Knowledge Distillation**: Temperature-based training
- **Real-World APIs**: Google Vision, AWS Rekognition, Custom APIs

---

## 📁 Proje Yapısı

```
model-stealer/
├── src/                          # Core kaynak kodları
│   ├── victim_model/            # Hedef model simülasyonu
│   ├── attacker/                # Saldırı stratejileri
│   ├── clone_model/             # Klon model eğitimi
│   └── utils/                   # Yardımcı araçlar
├── simulation/                   # 🔒 Güvenli eğitim ortamı
│   ├── config/                  # Simülasyon ayarları
│   ├── simulator.py             # Ana simülasyon scripti
│   └── README.md                # Simülasyon rehberi
├── real_world/                   # ⚠️ Gerçek API saldırıları (İZİNLİ)
│   ├── config/                  # Real-world ayarları
│   ├── adapters/                # API adaptörleri
│   ├── examples/                # API-specific örnekler
│   ├── real_world_attack.py     # Ana saldırı scripti
│   └── README.md                # Real-world rehberi
├── examples/                     # Örnek kullanımlar
├── docs/                        # Dokümantasyon
├── tests/                       # Test dosyaları
└── requirements.txt             # Gerekli paketler
```

---

## 🔧 Kurulum

```bash
# Projeyi klonla
git clone https://github.com/ertanszr-git/AI-Model-Stealer-Simulation.git
cd AI-Model-Stealer-Simulation

# Otomatik kurulum
chmod +x setup.sh
./setup.sh

# Manuel kurulum
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🔒 Mod 1: Güvenli Eğitim Simülatörü (Önerilen)

### 📚 Ne İçin Kullanılır?
- ✅ **Eğitim amaçlı**: Model extraction tekniklerini öğrenmek
- ✅ **Güvenli test**: Hiçbir yasal risk olmadan deneme
- ✅ **Araştırma**: Algorithm geliştirme ve test etme
- ✅ **Demo**: Tekniği gösterme ve anlatma

### 🚀 Nasıl Kullanılır?

#### Basit Kullanım:
```bash
# Simülasyon klasörüne git
cd simulation/

# Varsayılan ayarlarla çalıştır
python simulator.py
```

#### Konfigürasyon ile:
```bash
# Kendi ayarlarınızla çalıştır
python simulator.py --config config/simulation.yaml
```

#### Adım Adım Örnek:
```bash
# 1. Sanal ortamı aktive et
source venv/bin/activate  # macOS/Linux

# 2. Simülasyon klasörüne git
cd simulation/

# 3. Farklı stratejilerle deneyin
python simulator.py  # Varsayılan: ResNet18 vs ResNet18

# 4. Sonuçları inceleyin
ls -la ../experiments/simulation_*/
```

### ⚙️ Konfigürasyon Seçenekleri:

`simulation/config/simulation.yaml` dosyasını düzenleyerek:

```yaml
# Hedef model değiştir
victim_model:
  type: "vgg16"        # resnet18, resnet50, vgg16
  dataset: "cifar100"  # cifar10, cifar100, imagenet
  num_classes: 100

# Saldırı stratejisi seç
attack:
  strategy: "active_learning"  # random, active_learning, adversarial
  query_budget: 5000          # Kaç sorgu yapılacak
  batch_size: 64              # Batch boyutu

# Klon model mimarisi
clone_model:
  architecture: "lightweight_cnn"  # Farklı mimari dene
  training_epochs: 100              # Daha uzun eğitim
```

### 📊 Beklenen Çıktılar:
```
📁 experiments/simulation_baseline_extraction_2024_07_24_10_30/
├── 📊 results/
│   └── experiment_results.json    # Fidelity: %95.2, Queries: 10000
├── 🤖 models/
│   └── clone_model.pth           # Eğitilmiş klon model
├── 📈 logs/
│   └── simulation.log            # Detaylı işlem logları
└── 💾 data/
    ├── stolen_queries.npy        # Çalınan sorgu verileri
    └── stolen_responses.npy      # API yanıtları
```

---

## ⚠️ Mod 2: Real-World API Saldırıları

### 🚨 YASAL UYARI
**Bu mod sadece izniniz olan API'lerde kullanılabilir!**
- ⚖️ İzinsiz kullanım **yasadışıdır**
- 💼 Yasal sorumluluk **tamamen sizde**
- 📝 **Yazılı izin** almalısınız

### 🎯 Ne İçin Kullanılır?
- 🔬 **Güvenlik araştırması**: Bug bounty programları
- 🛡️ **Kendi API'nizi test**: Savunma geliştirme
- 🎓 **Akademik çalışma**: İzinli araştırma projeleri
- 🔍 **Penetrasyon testi**: Red team egzersizleri

### 📋 Önce Yapmanız Gerekenler:

#### 1. Yasal İzin Alın:
```markdown
☐ API sahibinden yazılı izin aldım
☐ Terms of Service'i okudum
☐ Test kapsamını belirledim  
☐ Hukuki danışmanlık aldım
☐ Sorumlu açıklama planım hazır
```

#### 2. API Bilgilerini Hazırlayın:
```yaml
# real_world/config/real_world.yaml
target_api:
  endpoint: "https://your-api.com/predict"
  api_key: "your-api-key"
  format: "json"  # veya "google_vision", "aws_rekognition"
```

### 🚀 Adım Adım Kullanım:

#### Adım 1: Güvenli Analiz
```bash
cd real_world/

# API'yi analiz et (saldırı yok, sadece keşif)
python real_world_attack.py --analyze-only
```

**Çıktı örneği:**
```
🔍 Target Analysis Results:
- Rate limits: 1000/hour
- Input format: base64 images
- Response format: JSON with predictions
- Estimated cost: $0.001 per request
```

#### Adım 2: Test Konfigürasyonu
```bash
# Konfigürasyonu test et
python real_world_attack.py --config config/real_world.yaml --analyze-only
```

#### Adım 3: Küçük Pilot Test
```yaml
# Önce küçük budget ile test
attack:
  query_budget: 100    # Sadece 100 sorgu
  batch_size: 1        # Tek tek sorgu
  strategy: "random"   # Basit strateji
```

#### Adım 4: Tam Saldırı (İzin gerekli!)
```bash
# Gerçek saldırı - DİKKATLİ!
python real_world_attack.py --config config/real_world.yaml
```

### 🎯 Desteklenen API'ler:

#### Google Cloud Vision API:
```yaml
target_api:
  endpoint: "https://vision.googleapis.com/v1/images:annotate"
  api_key: "your-google-api-key"
  format: "google_vision"
  cost_per_request: 0.0015
```

#### AWS Rekognition:
```yaml
target_api:
  format: "aws_rekognition" 
  aws_access_key: "AKIA..."
  aws_secret_key: "secret..."
  region: "us-east-1"
  cost_per_request: 0.001
```

#### Custom REST API:
```yaml
target_api:
  endpoint: "https://my-api.com/classify"
  headers:
    "Authorization": "Bearer token"
    "Content-Type": "application/json"
  format: "json"
  input_format: "base64"
```

### 📊 Örnek Real-World Sonuçları:
```
🎯 Google Vision API Attack Results:
- Fidelity: 87.3%
- Queries used: 2,847/5,000
- Total cost: $4.27
- Success rate: 94.2%
- Attack duration: 2h 15m

⚠️ Recommendations:
- Model shows vulnerability to extraction
- Consider implementing query budgets
- Add differential privacy noise
```

### 🛡️ Güvenlik Özellikleri:

#### Rate Limiting:
```yaml
# Etik kullanım için
rate_limit_delay: 2.0      # 2 saniye bekleme
randomize_delays: true     # Rastgele gecikme
max_total_cost: 50.0       # Maksimum $50
```

#### Stealth Mode:
```yaml
stealth:
  user_agent_rotation: true
  temporal_distribution: true
  request_randomization: true
```

---

## 📊 Karşılaştırma Tablosu

| Özellik | 🔒 Simülasyon | ⚠️ Real-World |
|---------|---------------|---------------|
| **Güvenlik** | ✅ Tamamen güvenli | 🚨 Yasal risk var |
| **Öğrenim** | ✅ Mükemmel | ⚠️ Dikkatli |
| **Maliyet** | ✅ Ücretsiz | 💰 API maliyeti |
| **Hız** | ✅ Çok hızlı | ⏱️ Rate limited |
| **Gerçeklik** | ⚠️ Simülasyon | ✅ Gerçek veriler |
| **İzin** | ✅ Gerek yok | 🚨 Mutlaka gerekli |

## 🏆 Hangi Modu Seçmeli?

### 🎓 Öğrenim/Eğitim için:
**➡️ Simülasyon modunu seçin**
- Hiçbir risk yok
- Hızlı sonuç
- Sınırsız deneme

### 🔬 Araştırma için:
**➡️ Önce simülasyon, sonra real-world**
1. Simülasyonda algoritmanızı geliştirin
2. Yasal izin alın  
3. Real-world'de test edin

### 🛡️ Güvenlik testi için:
**➡️ Real-world (kendi API'nız)**
- Kendi API'nizi test edin
- Gerçek vulnerabilityler keşfedin
- Savunma geliştirin

---

```bash
# Projeyi klonla
git clone https://github.com/ertanszr-git/AI-Model-Stealer-Simulation.git
cd AI-Model-Stealer-Simulation

# Otomatik kurulum
chmod +x setup.sh
./setup.sh

# Manuel kurulum
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🔧 Teknik Özellikler

### ✅ Tamamlanan Özellikler

- **Kurban Model Simülasyonu**: ResNet, VGG gibi modern mimariler
- **Saldırı Stratejileri**: 
  - Rastgele sorgulama
  - Aktif öğrenme tabanlı
  - Adversarial sorgulama
- **Klon Model Eğitimi**: Knowledge Distillation ile
- **Değerlendirme Metrikleri**:
  - Model Fidelity (Uyum oranı)
  - Query Efficiency (Sorgu verimliliği)
  - Cosine Similarity
- **Savunma Mekanizmaları**:
  - Rate limiting
  - Output noise injection
  - Query monitoring

### 🔧 Teknik Detaylar

**Desteklenen Mimariler:**
- ResNet (18, 50), VGG16, Custom CNN modeller

**Veri Setleri:**
- CIFAR-10/100, Custom datasets

**Saldırı Stratejileri:**
- Random queries, Active learning, Adversarial queries

## 📊 Örnek Sonuçlar

```
📊 Experiment Summary:
Victim Model Accuracy: 0.9234
Clone Model Accuracy: 0.8567
Model Fidelity: 0.8934
Queries Used: 2,000
Query Efficiency: 4.467 (fidelity per 1000 queries)
```

## 🛡️ Güvenlik ve Savunma

Proje ayrıca şu savunma mekanizmalarını simüle eder:

1. **API Rate Limiting**: Sorgu hızı sınırlama
2. **Output Perturbation**: Çıktılara noise ekleme
3. **Query Pattern Detection**: Anormal sorgu tespiti
4. **Differential Privacy**: Gizlilik koruyucu teknikler

## 📊 Performans Beklentileri

### 🔒 Simülasyon Modu:
- **Hız**: 1000-5000 sorgu/dakika
- **Fidelity**: %85-98 (tipik)
- **Süre**: 5-30 dakika
- **Maliyet**: $0

### ⚠️ Real-World Modu:
- **Hız**: 10-100 sorgu/dakika (rate limit)
- **Fidelity**: %70-95 (API'ye göre)
- **Süre**: 1-10 saat
- **Maliyet**: $1-100 (budget'a göre)

## 🔧 Troubleshooting

### ❌ Sık Karşılaşılan Sorunlar:

#### 1. Import Hataları:
```bash
# Çözüm: Virtual environment aktive edin
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. CUDA/MPS Hataları:
```bash
# Çözüm: Config'de device ayarı
device: "cpu"  # veya "mps" (Mac), "cuda" (GPU)
```

#### 3. API Connection Failed:
```bash
# Çözüm: API key ve endpoint kontrol
python real_world_attack.py --analyze-only  # Test connection
```

#### 4. Rate Limit Exceeded:
```yaml
# Çözüm: Config'de rate limit artır
rate_limit_delay: 5.0  # 5 saniye bekleme
batch_size: 1          # Tek sorgu
```

#### 5. Low Fidelity Results:
```yaml
# Çözüm: Daha fazla sorgu veya better strategy
attack:
  query_budget: 10000
  strategy: "active_learning"
  temperature: 3.0
```

#### 6. Memory Issues:
```yaml
# Çözüm: Batch size'ı azalt
clone_model:
  batch_size: 16  # Varsayılan 64'ten düşür
```

### 🆘 Destek Almak:

1. **Teknik Sorunlar**: GitHub Issues
2. **Yasal Sorular**: Hukuk müşaviri  
3. **Etik Konular**: Araştırma topluluğu
4. **API Problems**: İlgili API dokümantasyonu

## 📈 İleri Seviye Kullanım

### Custom Strategy Geliştirme:
```python
# src/attacker/custom_strategy.py
from .extraction_strategies import QueryStrategy

class MyCustomStrategy(QueryStrategy):
    def select_queries(self, pool, budget):
        # Kendi algoritmanız
        return selected_queries
```

### Multi-Target Attacks:
```bash
# Birden fazla hedef test et
for api in google_vision aws_rekognition azure_vision; do
    python real_world_attack.py --config config/${api}.yaml
done
```

### Ensemble Attacks:
```yaml
# Birden fazla stratejinin kombinasyonu
attack:
  strategy: "ensemble"
  strategies: ["random", "active_learning", "adversarial"]
  weights: [0.3, 0.5, 0.2]
```

## 📚 Eğitim Amaçları

Bu simülatör şunları öğrenmeye yardımcı olur:

- Model extraction saldırılarının nasıl çalıştığı
- Farklı sorgulama stratejilerinin etkinliği
- ML sistemlerinde API güvenliğinin önemi
- Model theft'e karşı savunma yöntemleri
- Knowledge distillation ve transfer learning

## 🤝 Katkıda Bulunma

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 Lisans

Bu proje eğitim amaçlıdır. Lütfen SECURITY.md dosyasını okuyun.

## 📞 İletişim

Sorularınız için issue açabilir veya [contact] ile iletişime geçebilirsiniz.

---

**⚠️ Hatırlatma**: Bu araç yalnızca eğitim ve araştırma amaçlıdır. Etik kurallara uygun kullanın!
