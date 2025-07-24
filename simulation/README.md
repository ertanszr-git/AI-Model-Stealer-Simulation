# 🔒 Simülasyon Modülü - Güvenli Eğitim Ortamı

Bu modül **tamamen güvenli** bir eğitim ve test ortamıdır. Hiçbir gerçek API'ye saldırı yapılmaz.

## 🎯 Ne Yapar?

- ✅ **Local model simulation**: Sadece yerel modeller kullanır
- ✅ **Safe learning environment**: Öğrenim için güvenli ortam
- ✅ **No external connections**: Dış bağlantı yok
- ✅ **Educational purpose**: Eğitim amaçlı tasarlanmış

## 🚀 Nasıl Kullanılır?

### Hızlı Başlangıç:
```bash
cd simulation/
python simulator.py
```

### Özel Config ile:
```bash
python simulator.py --config config/simulation.yaml
```

## 📁 Dosya Yapısı

```
simulation/
├── config/
│   └── simulation.yaml        # Simülasyon ayarları
├── simulator.py               # Ana simülasyon scripti
└── README.md                  # Bu dosya
```

## ⚙️ Konfigürasyon

`config/simulation.yaml` dosyasını düzenleyerek:

- **Victim Model**: Hedef model türü (ResNet, VGG, vb.)
- **Attack Strategy**: Saldırı stratejisi (random, active_learning, vb.)
- **Query Budget**: Sorgu bütçesi
- **Clone Architecture**: Klon model mimarisi

## 📊 Çıktılar

Simülasyon sonunda şunları alırsınız:

- **Fidelity Score**: Klon modelin başarı oranı
- **Extraction Statistics**: Çıkarma istatistikleri
- **Trained Clone Model**: Eğitilmiş klon model
- **Detailed Logs**: Detaylı loglar

## 🎓 Eğitim Amaçları

Bu modül şunları öğretir:

1. **Model Extraction Techniques**: Model çıkarma teknikleri
2. **Attack Strategies**: Farklı saldırı stratejileri
3. **Defense Mechanisms**: Savunma mekanizmaları
4. **ML Security**: ML güvenlik prensipleri

## ✅ Güvenlik Garantileri

- 🔒 **No real APIs attacked**: Gerçek API'lere saldırı yok
- 🔒 **No external connections**: Dış bağlantı yok
- 🔒 **Local computation only**: Sadece yerel hesaplama
- 🔒 **Safe for all environments**: Tüm ortamlarda güvenli

---

**📚 Not**: Gerçek dünya saldırıları için `../real_world/` klasörüne bakın (sadece izinli testler için).
