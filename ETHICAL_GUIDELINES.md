# 🛡️ Ethical Guidelines & Legal Compliance

## ⚠️ Kritik Uyarı

Bu AI Model Extraction Simulator **sadece eğitim ve güvenlik araştırması** amacıyla geliştirilmiştir. Gerçek dünya kullanımında aşağıdaki kurallara **mutlaka** uymanız gerekir.

## 📋 Yasal Gereklilikler

### 1. İzin ve Yetkilendirme
- ✅ **Test izni alın**: Hedef API'nin sahibinden açık izin alın
- ✅ **Yazılı anlaşma**: Mümkünse yazılı test anlaşması yapın
- ✅ **Kapsam belirleyin**: Hangi testlerin yapılacağını önceden bildirin
- ❌ **İzinsiz test yasaktır**: Hiçbir zaman izinsiz API'leri test etmeyin

### 2. Terms of Service (ToS) Uyumluluğu
```yaml
# Kontrol edilmesi gerekenler:
checklist:
  - API kullanım şartlarını okudunuz mu?
  - Rate limit kurallarına uyuyor musunuz?
  - Veri kullanım politikalarına uygun mu?
  - Ters mühendislik yasağı var mı?
```

### 3. Veri Gizliliği
- 🔒 **Kişisel veri kullanmayın**: Gerçek kişilere ait görüntü kullanmayın
- 🔒 **Sentetik veri kullanın**: Rastgele/sentetik görüntüler tercih edin
- 🔒 **Veri saklama**: Gereksiz veri saklamayın
- 🔒 **Güvenli silme**: Test sonrası verileri güvenli şekilde silin

## 🎯 Responsible Disclosure

### Güvenlik Açığı Keşfetmeniz Durumunda:

1. **Hemen durdurun**: Saldırıyı derhal sonlandırın
2. **Dokümante edin**: Keşfedilen açığı detaylıca kaydedin
3. **Vendor'a bildirin**: API sağlayıcısına sorumlu şekilde bildirin
4. **Süre tanıyın**: Düzeltme için makul süre (90-120 gün) tanıyın
5. **Koordineli açıklama**: Vendor ile koordineli şekilde açıklayın

### Raporlama Şablonu:
```markdown
# Güvenlik Açığı Raporu

## Özet
- API: [API Adı]
- Açık Türü: Model Extraction Vulnerability
- Ciddiyet: [Düşük/Orta/Yüksek/Kritik]

## Teknik Detaylar
- Saldırı Vektörü: [Detaylar]
- Başarı Oranı: [%]
- Gereken Sorgu Sayısı: [Sayı]
- Maliyet: [$]

## Etki
- [Potansiyel etkileri açıklayın]

## Çözüm Önerileri
- [Düzeltme önerileriniz]
```

## 🚨 Yasadışı Kullanım Uyarıları

### Kesinlikle YAPMAYIN:
- ❌ İzinsiz ticari API'leri saldırmayın
- ❌ Rekabet avantajı için model çalmayın
- ❌ Telif hakkı ihlali yapmayın
- ❌ Kişisel veri çalmayın
- ❌ DoS (Denial of Service) saldırısı yapmayın
- ❌ Yasadışı içerik üretmek için kullanmayın

### Olası Yasal Sonuçlar:
```
🏛️ Ceza Hukuku:
   - Bilgisayar dolandırıcılığı
   - Yetkisiz erişim
   - Veri hırsızlığı

💼 Hukuki Sorumluluk:
   - Telif hakkı ihlali
   - Ticari sır çalma
   - Rekabet hukuku ihlali

💰 Mali Yaptırımlar:
   - Tazminat ödeme
   - Lisans ücretleri
   - Yasal masraflar
```

## 🛡️ Güvenli Kullanım Rehberi

### 1. Test Ortamları
```yaml
# Güvenli test hedefleri:
safe_targets:
  - localhost: "Kendi API'nız"
  - test_apis: "Özel test API'leri"
  - open_source: "Açık kaynak modeller"
  - research_apis: "Araştırma amaçlı API'ler"

# RİSKLİ hedefler:
risky_targets:
  - production_apis: "Canlı ticari API'ler"
  - third_party: "Üçüncü taraf servisleri"
  - unknown_ownership: "Sahipliği belirsiz API'ler"
```

### 2. Rate Limiting Best Practices
```python
# Etik rate limiting
ethical_limits = {
    'requests_per_second': 0.1,  # Çok yavaş
    'requests_per_minute': 5,    # Maksimum 5/dakika
    'requests_per_hour': 100,    # Maksimum 100/saat
    'requests_per_day': 1000     # Maksimum 1000/gün
}

# Rastgele gecikmeler
randomized_delays = {
    'min_delay': 2.0,   # En az 2 saniye
    'max_delay': 10.0,  # En fazla 10 saniye
    'exponential_backoff': True
}
```

### 3. Stealth ve Nezaket
```python
stealth_config = {
    # Nezaketi temsil eden özellikler
    'user_agent_rotation': True,
    'request_randomization': True,
    'temporal_distribution': True,
    
    # Agresif OLMAYAN özellikler
    'no_concurrent_requests': True,
    'respect_http_status_codes': True,
    'honor_retry_after_headers': True
}
```

## 📚 Eğitim ve Araştırma Kullanımı

### Uygun Kullanım Alanları:
- ✅ **Akademik araştırma**: Üniversite projesi
- ✅ **Güvenlik araştırması**: Bug bounty programları
- ✅ **Eğitim amaçlı**: Siber güvenlik eğitimi
- ✅ **Self-assessment**: Kendi API'nizi test etme
- ✅ **Red team exercises**: İzinli penetrasyon testleri

### Araştırma Etiği:
```markdown
1. IRB Onayı: İnsan denekli araştırmalarda IRB onayı alın
2. Peer Review: Çalışmanızı meslektaşlarınıza gösterin
3. Şeffaflık: Metodolojinizi açıkça paylaşın
4. Reproducibility: Sonuçlarınızın tekrarlanabilir olmasını sağlayın
```

## 🔍 Monitoring ve Detection Avoidance

### Etik Monitoring
```python
# Kendi aktivitenizi monitoring edin
monitoring = {
    'log_all_requests': True,
    'track_costs': True,
    'measure_impact': True,
    'document_findings': True
}

# API provider'ın detection sistemlerine saygı gösterin
respect_detection = {
    'stop_when_blocked': True,
    'respect_captchas': True,
    'honor_ip_bans': True,
    'reduce_rate_when_warned': True
}
```

## 💡 Konstruktif Feedback

### API Sağlayıcılarına Öneriler:
```python
# Güvenlik iyileştirmeleri
security_recommendations = {
    'rate_limiting': 'Adaptif rate limiting',
    'query_analysis': 'Anormal sorgu pattern detection',
    'model_fingerprinting': 'Model benzersizlik koruması',
    'differential_privacy': 'Noise injection',
    'query_budgets': 'Kullanıcı başına sorgu limiti'
}
```

## 📞 İletişim ve Destek

### Sorun Bildirimi:
```
🚨 Yasal Sorunlar:
   - Hukuk müşavirinize danışın
   - Yerel yasaları kontrol edin
   - Uluslararası düzenlemeleri gözden geçirin

🔧 Teknik Sorunlar:
   - GitHub Issues kullanın
   - Detaylı hata raporu yazın
   - Reproducible examples sağlayın

📧 Etik Sorular:
   - Araştırma topluluğuna danışın
   - Etik komitelerle iletişime geçin
   - Expert review talep edin
```

## 🏛️ Yasal Çerçeve Örnekleri

### ABD Yasaları:
- **Computer Fraud and Abuse Act (CFAA)**
- **Digital Millennium Copyright Act (DMCA)**
- **Trade Secrets Act**

### AB Düzenlemeleri:
- **GDPR (Veri Koruma)**
- **Computer Crime Directive**
- **Trade Secrets Directive**

### Türkiye Yasaları:
- **5237 sayılı TCK (Bilişim Suçları)**
- **6698 sayılı KVKK**
- **Fikri ve Sınai Haklar**

## ✅ Son Kontrol Listesi

Gerçek dünya kullanımından önce:

```markdown
☐ Yasal izin aldım
☐ ToS'u okudum ve uygun
☐ Rate limit'leri ayarladım
☐ Veri gizliliği kontrolü yaptım
☐ Emergency stop planım hazır
☐ Sorumlu açıklama planım var
☐ Backup/rollback stratejim mevcut
☐ Cost budget'ım belirli
☐ Monitoring sistemim aktif
☐ Hukuki danışmanlık aldım
```

---

**⚠️ ÖNEMLİ**: Bu rehber genel bir kılavuzdur ve yasal tavsiye değildir. Gerçek hukuki durumlar için mutlaka nitelikli bir hukuk müşavirine başvurun.

**🏛️ YASAL UYARI**: Bu araçları kullanarak tüm yasal sorumlulukları kabul etmiş sayılırsınız. Geliştiriciler hiçbir yasal sorumluluk kabul etmez.
