# Bölüm 2: CNN Mimarisi, Encoding Yöntemleri ve İlk Zaman-Serisi Denemesi

**Proje:** IMEP - Nöromorfik Hızlandırıcılarla Kestirimci Bakım
**Framework:** SpikingJelly
**Devam:** [Bölüm 1 — SNN Temelleri](01-snn-fundamentals.md)

---

## 1. CNN Tabanlı SNN (MNIST Üzerinde)

### Amaç

Bölüm 1'de tam-bağlantılı (Linear) bir SNN ile MNIST'te %96.39 doğruluk elde etmiştik. Bu kez **Convolutional (evrişimli)** bir mimari deneyerek, yerel örüntü yakalamanın (kenar, köşe gibi) performansa etkisini gözlemledik.

### Mimari

```
Görüntü (1x28x28)
  → Conv2d(1→16, 3x3) → LIF → MaxPool2d(2)   [16x14x14]
  → Conv2d(16→32, 3x3) → LIF → MaxPool2d(2)  [32x7x7]
  → Flatten → Linear(1568→10) → LIF
```

### Sonuçlar

| Epoch | Loss   | Test Doğruluğu |
|-------|--------|-----------------|
| 1     | 0.0274 | 96.91%          |
| 2     | 0.0067 | 97.83%          |
| 3     | 0.0051 | 98.15%          |

### Karşılaştırma

| Model | Epoch 3 Doğruluk |
|-------|-------------------|
| Linear SNN (Bölüm 1) | 96.39% |
| Conv SNN | **98.15%** |

**Çıkarım:** CNN mimarisi, görüntüdeki yerel örüntüleri yakalayabildiği için Linear modele göre daha yüksek doğruluk sağladı. Bu, kestirimci bakım verisinde (zaman serisi sinyallerindeki yerel örüntüleri yakalamak için) Conv1d tabanlı bir mimariye yönelmenin mantıklı olacağını gösteriyor.

---

## 2. Encoding Yöntemleri Karşılaştırması

Aynı sinyal (0-1 arası normalize sinüs dalgası, 50 zaman adımı) üç farklı encoding yöntemiyle spike dizisine çevrildi.

### 2.1 Poisson (Rate) Encoding

`encoding.PoissonEncoder()`. Sinyal değeri, o adımda spike üretme **olasılığını** belirler.
- Sinyalin yüksek olduğu bölgelerde spike yoğunluğu arttı, düşük bölgelerde azaldı.
- **Olasılıksal**: aynı sinyal tekrar verilse farklı bir spike paterni çıkabilir.

### 2.2 Latency Encoding

`encoding.LatencyEncoder(T=time_steps)`. Bilgi, **ilk spike'ın zamanında** taşınır.

| Genlik | İlk Spike Zamanı |
|--------|-------------------|
| 0.98 (yüksek) | ~1 (çok erken) |
| 0.56 (orta) | ~21 (orta) |
| 0.07 (düşük) | ~45 (çok geç) |

Genlik yüksekse spike erken, düşükse geç gelir. Çok az sayıda spike ile bilgi taşıyabildiği için **enerji açısından en verimli yöntem**.

### 2.3 Direct Encoding

Ham sinyal, encoder kullanılmadan doğrudan bir LIF nörona verildi.
- İlk denemede eşik (1.0) sinyalin maksimum değerinden yüksek olduğu için hiç spike üretilmedi (LIF satürasyon davranışı). Eşik `0.4`'e düşürülünce 19 spike üretildi, sinyalin iki tepe noktasına denk gelen kümelerde yoğunlaştı.
- **Deterministik**: aynı sinyal tekrar verilirse birebir aynı spike paterni çıkar (Poisson'dan farkı budur).

### Genel Karşılaştırma

| Yöntem | Bilgi Taşıma Şekli | Enerji Verimliliği | Kullanım Senaryosu |
|--------|----------------------|----------------------|----------------------|
| Poisson (Rate) | Spike sıklığı | Orta-yüksek | Gürültüye dayanıklı, genel amaçlı |
| Latency | İlk spike zamanı | Çok düşük | Hızlı tepki / enerji kritik senaryolar |
| Direct | LIF'in kendi dinamiği (deterministik) | Ayarlanabilir (eşiğe bağlı) | Basit prototipleme, manuel eşik ayarı gerekir |

**Kestirimci bakım için çıkarım:** Latency encoding, enerji verimliliği açısından en güçlü aday — "anormallik ne kadar şiddetliyse o kadar hızlı alarm ver" mantığıyla örtüşüyor. Ancak gürültüye karşı dayanıklılığı gerçek veri üzerinde test edilmeli.

---

## 3. Sentetik Zaman-Serisi Verisiyle İlk SNN Sınıflandırıcı

### Amaç

Gerçek kestirimci bakım verisetine geçmeden önce, framework'ün zaman-serisi veriyle nasıl çalıştığını görmek için kendi sentetik verimizi ürettik.

### Veri Üretimi

- **Normal sınıf:** Düşük genlikli, düzenli sinüs + az gürültü (sağlıklı makine benzetmesi)
- **Arızalı sınıf:** Aynı temel sinyal + rastgele ani sıçramalar + daha fazla gürültü (arızalı rulman/motor benzetmesi)
- 200 normal + 200 arızalı örnek, her biri 100 zaman adımı uzunluğunda

### Model Mimarisi (Conv1D SNN)

```
Sinyal (1x100)
  → Conv1d(1→8, kernel=5) → LIF → MaxPool1d(2)   [8x50]
  → Conv1d(8→16, kernel=5) → LIF → MaxPool1d(2)  [16x25]
  → Flatten → Linear(400→2) → LIF   (2 sınıf: normal/arızalı)
```

MNIST'teki Conv2d'nin 1 boyutlu (zaman ekseni) versiyonu olan **Conv1d** kullanıldı.

### İlk Eğitim Sonuçları (lr=1e-3)

| Epoch | Loss   | Test Doğruluğu |
|-------|--------|-----------------|
| 1-6   | 0.5000 | 50.00% (donuk)  |
| 7     | 0.4063 | 55.00%          |
| 10    | 0.2147 | 86.25%          |
| 14    | 0.1879 | 95.00%          |
| 15    | 0.1809 | **95.00%**      |

### Debug: "Donukluk" Problemi ve Learning Rate Deneyi

İlk 6 epoch boyunca model hiçbir şey öğrenmedi (Loss=0.5, Doğruluk=%50 — rastgele tahmin seviyesi). Bu, MNIST'te de karşılaşılan "eğitilmemiş ağırlıklar eşiği geçemiyor" probleminin bir varyasyonu: nöronlar ateşlemediği sürece gradyan akışı zayıf kalıyor, ağırlıklar öğrenemiyor ("ölü nöron" benzeri durum).

**Çözüm denemesi:** Öğrenme oranı `1e-3`'ten `5e-3`'e çıkarıldı.

| | lr=1e-3 | lr=5e-3 |
|---|---|---|
| Donukluk süresi | 6 epoch | **3 epoch** |
| En yüksek doğruluk | 95.00% | **96.25%** |
| Son epoch loss | 0.1809 | **0.0570** |

**Çıkarım:** Öğrenme oranını artırmak, ağırlıkların eşiği geçecek seviyeye daha hızlı ulaşmasını sağladı. Gerçek kestirimci bakım verisine ve donanıma (Akida) geçildiğinde, benzer bir donukluk gözlemlenirse öğrenme oranı artırımı denenmesi gereken ilk çözümlerden biri olmalı.

---

## 4. Genel Özet — Bugün Öğrenilenler

- Conv1d/Conv2d'nin Linear katmanlara göre yerel örüntü yakalama avantajı
- Üç farklı encoding yöntemi (Poisson, Latency, Direct) ve enerji/kullanım açısından farkları
- Kendi sentetik zaman-serisi verisini üretme (PyTorch `Dataset` sınıfı ile)
- Zaman-serisi veri için Conv1D tabanlı SNN mimarisi kurma
- SNN eğitiminde "donukluk" (dead neuron benzeri) problemi ve learning rate ile debug etme

## 5. Sıradaki Adımlar

- [ ] Gerçek kestirimci bakım veriseti seçimi (CWRU, NASA CMAPSS gibi adaylar)
- [ ] Gerçek veri üzerinde encoding stratejisi seçimi (muhtemelen Latency veya Poisson denenecek)
- [ ] Donanım kurulumları (Akida kartları, Jetson)

---

*Bu doküman, IMEP kapsamında yürütülen "Nöromorfik Hızlandırıcılarla Kestirimci Bakım" çalışmasının ikinci bölüm ilerleme notlarını içermektedir.*
