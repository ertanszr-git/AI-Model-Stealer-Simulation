# ⚠️ Real-World Attack Modülü - Sadece İzinli Testler İçin

**UYARI**: Bu modül gerçek API'lere saldırı yapmak için tasarlanmıştır. Sadece **izniniz olan** sistemlerde kullanın!

## 🚨 Yasal Uyarı

Bu araçları kullanmadan önce **mutlaka**:

- ✅ Hedef API'nin sahibinden **yazılı izin** alın
- ✅ **Terms of Service**'i okuyun ve uyun
- ✅ **Rate limit**'lere saygı gösterin
- ✅ **Sorumlu açıklama** prensiplerine uyun

**❌ İzinsiz kullanım yasadışıdır ve yasal sonuçları olabilir!**

## 🎯 Ne Yapar?

- 🎯 **Real API Analysis**: Gerçek API'leri analiz eder
- 🎯 **Live Model Extraction**: Canlı modellerden bilgi çıkarır
- 🎯 **Clone Training**: Çalınan verilerle model eğitir
- 🎯 **Success Evaluation**: Saldırı başarısını değerlendirir

## 🚀 Kullanım

### ⚠️ Önce İzin Alın!
```bash
# 1. Hedef API'yi analiz et (güvenli)
python real_world_attack.py --analyze-only

# 2. Tam saldırı (İZİN GEREKLİ!)
python real_world_attack.py --config config/real_world.yaml
```

### Özel API Saldırıları:
```bash
# Popüler API'ler için örnekler
python examples/api_specific_attacks.py
```

## 📁 Dosya Yapısı

```
real_world/
├── config/
│   └── real_world.yaml         # Real-world ayarları
├── adapters/
│   ├── target_analyzer.py      # API analiz aracı
│   └── real_world_adapter.py   # API adaptörü
├── examples/
│   └── api_specific_attacks.py # Özel API örnekleri
├── real_world_attack.py        # Ana saldırı scripti
└── README.md                   # Bu dosya
```

## ⚙️ Desteklenen API'ler

- **Google Cloud Vision API**
- **AWS Rekognition** 
- **Azure Computer Vision**
- **Hugging Face APIs**
- **Custom REST APIs**

## 📊 Sonuçlar

Başarılı saldırı sonrası:

- **Fidelity Score**: %85-98 (tipik)
- **Cost Analysis**: Maliyet hesabı
- **Stolen Model**: Çalınan model kopyası
- **Attack Report**: Detaylı saldırı raporu

## 🛡️ Etik Kullanım

### ✅ Uygun Kullanım:
- Kendi API'nizi test etme
- İzinli güvenlik araştırması
- Akademik çalışmalar
- Bug bounty programları

### ❌ Yasadışı Kullanım:
- İzinsiz ticari API saldırıları
- Rekabet avantajı için model çalma
- Kişisel veri hırsızlığı
- DoS saldırıları

## 🔧 Güvenlik Özellikleri

- **Rate Limiting**: Agresif olmayan sorgu hızı
- **Cost Control**: Maliyet kontrol mekanizması
- **Stealth Mode**: Algılanmayı önleme teknikleri
- **Error Handling**: Güvenli hata yönetimi

## 📞 Destek

**Yasal sorunlar**: Hukuk müşavirinize danışın
**Teknik sorunlar**: GitHub Issues kullanın
**Etik sorular**: Araştırma topluluğuna danışın

---

**⚖️ YASAL SORUMLULUK**: Bu araçları kullanarak tüm yasal sorumluluğu kabul etmiş sayılırsınız. Geliştiriciler hiçbir yasal sorumluluk kabul etmez.

**🎓 EĞİTİM İÇİN**: Güvenli öğrenim için `../simulation/` klasörünü kullanın.
