# Bölüm 3: Veriset Seçimi ve CWRU ile İlk Gerçek Veri Denemesi

**Proje:** IMEP - Nöromorfik Hızlandırıcılarla Kestirimci Bakım
**Framework:** SpikingJelly
**Devam:** [Bölüm 2 — CNN, Encoding, Sentetik Zaman Serisi](02-cnn-encoding-timeseries.md)

---

## 1. Veriset Seçimi: CWRU vs NASA CMAPSS

Olası kestirimci bakım verisetleri araştırıldı.

| | CWRU Bearing | NASA CMAPSS |
|---|---|---|
| Problem tipi | Sınıflandırma (arıza tipi tespiti) | Regresyon (kalan ömür/RUL tahmini) |
| Veri kaynağı | Gerçek fiziksel deney (titreşim sensörü) | Simülasyon |
| Format | `.mat` | `.txt` |
| Sinyal tipi | Tek boyutlu, yüksek frekans (12-48 kHz) | Çok boyutlu (21 sensör), döngü bazlı |

**Karar: CWRU Bearing Dataset** — mevcut sentetik veri denemesiyle (normal/arızalı sınıflandırma, tek kanallı sinyal) yapısal olarak örtüştüğü ve sınıflandırma probleminin SNN literatüründe daha yaygın olduğu için tercih edildi. Detaylı gerekçe: [datasets/dataset-selection.md](../datasets/dataset-selection.md)

---

## 2. CWRU Verisini İndirme ve Tanıma

CWRU Bearing Data Center'dan (12kHz drive-end verisi, 1797 rpm, 0.007" arıza boyutu) 4 dosya indirildi:

| Dosya | Sınıf | Ham Örnek Sayısı |
|-------|-------|---------------------|
| 97.mat | Normal | 243,938 |
| 105.mat | Inner Race | 121,265 |
| 118.mat | Ball | 122,571 |
| 130.mat | Outer Race | 121,991 |

Her `.mat` dosyası `DE_time` (Drive End), `FE_time` (Fan End) ve bazılarında `BA_time` (Base) değişkenlerini içeriyor. Sınıflandırma için `DE_time` kullanıldı.

### Görsel Gözlem

Ham sinyaller görselleştirildiğinde her sınıfın kendine özgü bir "imza" olduğu görüldü:
- **Normal:** Düşük genlik (±0.2), düzenli/periyodik salınım.
- **Inner Race:** Genlik ~5 kat arttı (±1.5), düzenli aralıklarla keskin darbeler.
- **Ball:** Sürekli ve yoğun gürültü karakteri, net tekil darbelerden çok karmaşık temas paterni.
- **Outer Race:** En yüksek genlik (±2.5), çok düzenli ve belirgin periyodik darbeler (en "temiz" arıza imzası).

**Çıkarım:** Gerçek arıza sinyalleri, önceki sentetik veri denemesindeki rastgele/seyrek sıçramalardan farklı olarak, düzenli ve periyodik darbe paternleri içeriyor.

---

## 3. Veri Hazırlama: Segmentleme (Pencereleme)

### İlk Deneme: Üst Üste Binmeyen Pencereler (Non-overlapping)

Ham sinyaller 1024 örneklik pencerelere bölündü, sınıf dengesizliği (Normal dosyası diğerlerinin ~2 katı uzunlukta) fark edilip undersampling ile dengelendi.

| Sınıf | Segment (dengeleme sonrası) |
|-------|-------------------------------|
| Normal | 119 |
| Inner Race | 118 |
| Ball | 119 |
| Outer Race | 119 |

**Toplam: 475 örnek** — sentetik veri denemesine (400 örnek) yakın bir büyüklük.

### İlk Eğitim Sonucu (475 örnek, 4 sınıf)

Conv1D SNN mimarisi (sentetik veri modelinin 1024 uzunluk ve 4 sınıfa uyarlanmış hali) ile eğitim yapıldı. Sonuç **kararsız** çıktı:

```
Epoch 4:  %62.11  (en yüksek erken nokta)
Epoch 6:  %27.37  (neredeyse rastgele seviyeye geriledi)
...
Epoch 20: %38.95  (son epoch, en yüksekten çok uzak)
```

**Sorun teşhisi:** Öğrenme oranı denendi (5e-3 → 1e-3), ama kararsızlık sürdü — asıl sorunun öğrenme oranı değil, **veri setinin küçüklüğü** (sınıf başına ~95-119 örnek, 4 sınıflı bir problem için yetersiz) olduğu anlaşıldı.

### Çözüm: Üst Üste Binen Pencereler (Overlapping Windows)

Aynı ham veriden daha fazla segment çıkarmak için `stride=512` (window_size'ın yarısı, %50 örtüşme) kullanıldı — bu, veri artırma (data augmentation) yöntemi.

| Sınıf | Segment (dengeleme öncesi) | Segment (dengeleme sonrası) |
|-------|-------------------------------|-------------------------------|
| Normal | 475 | 235 |
| Inner Race | 235 | 235 |
| Ball | 238 | 235 |
| Outer Race | 237 | 235 |

**Toplam: 940 örnek** (önceki 475'in ~2 katı).

---

## 4. Final Eğitim Sonuçları (940 örnek)

Train: 752, Test: 188. Öğrenme oranı: 1e-3.

```
Epoch 1-5:   %28.19 (donukluk - beklenen başlangıç davranışı)
Epoch 7:     %65.43
Epoch 14:    %75.00
Epoch 19-24: %73-76 arası (nispeten istikrarlı bölge)
Epoch 21/23/24: %76.60 (en iyi)
```

**En iyi test doğruluğu: %76.60**

### Karşılaştırma

| | 475 örnek | 940 örnek |
|---|---|---|
| En iyi doğruluk | %74.74 | **%76.60** |
| İstikrar | Çok dalgalı | Epoch 19-24 arası nispeten kararlı |
| Son epoch loss | 0.1408 | **0.1293** |

**Çıkarım:** Daha fazla veri (overlapping window ile), hem doğruluğu artırdı hem de eğitimi kısmen daha istikrarlı hale getirdi. Tam istikrar hâlâ yok — bu, küçük veri setleriyle SNN eğitiminin doğasında olan bir zorluk olarak değerlendiriliyor. 4 sınıflı bir problemde rastgele tahmin ~%25 olduğu için, %76.60 makul bir ilk sonuç.

---

## 5. Genel Özet — Bugün Öğrenilenler

- CWRU ve NASA CMAPSS verisetlerinin karşılaştırılması ve gerekçeli seçim yapılması
- `.mat` dosyalarını `scipy.io.loadmat` ile okuma
- Ham sinyal görselleştirme ve sınıflar arası görsel farkları yorumlama
- Sinyal segmentleme (pencereleme): non-overlapping vs overlapping window stratejileri
- Sınıf dengesizliği tespiti ve undersampling ile düzeltme
- Küçük veri setlerinde SNN eğitiminin kararsızlık eğilimi ve veri artırmanın etkisi
- Gerçek veriyle uçtan uca bir SNN sınıflandırıcı kurma, eğitme, debug etme

## 6. Sıradaki Adımlar

- [ ] (Opsiyonel) Daha fazla veri artırma stratejisi (daha küçük stride, farklı motor yükü/hız verilerini ekleme)
- [ ] (Opsiyonel) Farklı encoding yöntemlerinin (Latency) CWRU verisinde denenmesi
- [ ] Donanım kurulumları (Akida kartları, Jetson)
- [ ] Diğer framework'lere geçiş 

---

*Bu doküman, IMEP kapsamında yürütülen "Nöromorfik Hızlandırıcılarla Kestirimci Bakım" çalışmasının üçüncü bölüm ilerleme notlarını içermektedir.*
