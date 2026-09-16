# Sentetik Zaman Serisi SNN - SpikingJelly
 
## Model: Conv1D SNN (Zaman Serisi Sınıflandırma)
 
Mimari: Conv1d(1→8) → LIF → MaxPool1d → Conv1d(8→16) → LIF → MaxPool1d → Linear(400→2) → LIF, T=10 zaman adımı.
 
### Veri Kümesi ve Batch Detayları
- **Train boyutu:** (320, 100)
- **Test boyutu:** (80, 100)
- **Bir batch örneği:**
  - X boyutu: `torch.Size([16, 100])`
  - y boyutu: `torch.Size([16])`
- **Model çıktı boyutu:** `torch.Size([1, 2])`

### Eğitim Sonuçları
 
| Epoch | Loss   | Test Doğruluğu |
|-------|--------|-----------------|
| 1     | 0.5000 | 50.00%          |
| 2     | 0.5000 | 50.00%          |
| 3     | 0.5000 | 50.00%          |
| 4     | 0.5000 | 50.00%          |
| 5     | 0.5000 | 50.00%          |
| 6     | 0.5000 | 50.00%          |
| 7     | 0.4063 | 55.00%          |
| 8     | 0.2548 | 50.00%          |
| 9     | 0.2482 | 55.00%          |
| 10    | 0.2147 | 86.25%          |
| 11    | 0.2027 | 82.50%          |
| 12    | 0.2003 | 81.25%          |
| 13    | 0.1856 | 78.75%          |
| 14    | 0.1879 | 95.00%          |
| 15    | 0.1809 | 95.00%          |
 
### Yorum
Eğitimin ilk 6 epoch'u boyunca doğruluk %50 (rastgele tahmin) düzeyinde kalmış ve kayıp (loss) 0.5000 seviyesinde sabitlenmiştir. 7. epoch'tan itibaren model öğrenmeye başlamış ve 15. epoch sonunda **%95.00** test doğruluğuna ulaşarak normal ve arızalı sinyalleri yüksek başarıyla sınıflandırmayı başarmıştır.

## Ek Not: Learning Rate Deneyi

İlk eğitimde (lr=1e-3) ilk 6 epoch boyunca model "donuk" kaldı (Loss=0.5, Doğruluk=%50, rastgele tahmin seviyesi). Öğrenme oranı `5e-3`'e çıkarılınca:

| | lr=1e-3 | lr=5e-3 |
|---|---|---|
| Donukluk süresi | 6 epoch | 3 epoch |
| En yüksek doğruluk | 95.00% | **96.25%** |
| Son epoch loss | 0.1809 | **0.0570** |

**Sebep:** Düşük öğrenme oranında, rastgele başlangıç ağırlıklarının LIF eşiğini geçecek seviyeye ulaşması yavaş kalıyor — nöronlar ateşlemediği sürece gradyan akışı zayıf kalıyor ("ölü nöron" benzeri bir durum). Öğrenme oranını artırmak bu geçişi hızlandırdı.

**Çıkarım:** Gerçek kestirimci bakım verisine ve donanıma (Akida) geçildiğinde, model eğitiminde ilk epoch'larda benzer bir "donukluk" gözlemlenirse, öğrenme oranını artırmak denenmesi gereken ilk çözümlerden biri olmalı.
