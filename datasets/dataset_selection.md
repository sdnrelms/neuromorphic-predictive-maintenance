# Veriset Seçimi: CWRU Bearing Dataset

## Karşılaştırılan Adaylar

| | CWRU Bearing | NASA CMAPSS |
|---|---|---|
| **Problem tipi** | Sınıflandırma (arıza tipi tespiti) | Regresyon (kalan ömür/RUL tahmini) |
| **Veri kaynağı** | Gerçek fiziksel deney (titreşim sensörü) | Simülasyon (sentetik ama gerçekçi) |
| **Format** | `.mat` (MATLAB) | `.txt` (düz metin) |
| **Sinyal tipi** | Tek boyutlu titreşim, yüksek frekans (12-48 kHz) | Çok boyutlu (21 sensör), döngü bazlı |
| **Zorluk** | Orta (net sınıflar) | Daha karmaşık (regresyon + çoklu sensör) |

## Karar: CWRU Bearing Dataset

**Gerekçe:**
1. Daha önce üretilen sentetik veri denemesiyle (normal vs arızalı sınıflandırma, tek kanallı sinyal) yapısal olarak örtüşüyor — mevcut Conv1D SNN mimarisi büyük değişiklik gerektirmeden uyarlanabilir.
2. Sınıflandırma problemi, SNN literatüründe regresyona göre daha yaygın ve daha iyi belgelenmiş.
3. `.mat` formatı `scipy.io.loadmat` ile kolayca okunabiliyor, ek bir engel oluşturmuyor.

**CMAPSS için not:** İleride vakit ve kapsam izin verirse, ikinci bir veriset olarak (daha karmaşık, regresyon tabanlı bir senaryo için) değerlendirilebilir.

## Kaynak

- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter
- Sınıflar: Normal (N), İç Bilezik Arızası (IR), Dış Bilezik Arızası (OR), Bilya Arızası (B)
- Örnekleme hızı: 12 kHz (drive-end verisi, en yaygın kullanılan)
