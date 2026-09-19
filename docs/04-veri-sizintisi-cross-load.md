# Bölüm 4: Veri Sızıntısı Analizi ve Çapraz-Yük Doğrulaması

**Proje:** IMEP - Nöromorfik Hızlandırıcılarla Kestirimci Bakım
**Framework:** SpikingJelly
**Devam:** [Bölüm 3 — Veriset Seçimi ve CWRU ile İlk Gerçek Veri Denemesi](03-cwru-real-data.md)

---

## 1. Confusion Matrix ile Gizli Sorunun Keşfi

Bölüm 3'teki %76.60 sonucunun ardında ne olduğunu anlamak için confusion matrix çıkarıldı. Sonuç beklenmedik bir sorunu ortaya çıkardı:

```
                 Tahmin:
              Normal  Inner  Ball  Outer
Gerçek Normal:   53     0     0     0
Gerçek Inner:    46     0     0     0
Gerçek Ball:     44     0     0     0
Gerçek Outer:    33     0     0    12
```

**Bulgu:** Model neredeyse her örneği "Normal" olarak tahmin ediyordu — yüksek görünen doğruluk, sınıf dengesizliğinden kaynaklanan bir yanılsamaydı (çoğunluk sınıfını seçerek bile yüksek doğruluk elde edilebiliyordu).

## 2. Loss Fonksiyonu Değişikliği: MSE → CrossEntropy

`MSELoss` yerine sınıflandırma için daha uygun olan `CrossEntropyLoss`'a geçildi. Ayrıca en iyi performans gösteren epoch'un model ağırlıklarını kaydetme (checkpointing) eklendi.

**Yeni sorun ortaya çıktı:** Loss, `ln(4) = 1.3863` değerinde donup kaldı — bu, modelin 4 sınıfa da eşit olasılık verdiği, hiç öğrenmediği anlamına geliyordu. Kök sebep: hiçbir nöron ateşlemiyordu (LIF eşiği hiç geçilemiyordu), bu yüzden gradyan akışı sıfır kalıyordu.

## 3. Mimari İyileştirmesi: BatchNorm ve Daha Geniş Kernel

Model mimarisi şu şekilde güncellendi:
- Kanal sayıları artırıldı (8→16, 16→32)
- Kernel boyutları büyütüldü (5→15, 5→11)
- Her Conv1d katmanından sonra **BatchNorm1d** eklendi
- LIF eşiği 0.5'e ayarlandı

Bu değişiklikle model **%100 test doğruluğuna** ulaştı — ancak bu sonuç şüpheyle karşılandı, çünkü SNN'lerde kusursuz doğruluk nadir görülür.

## 4. Veri Sızıntısı Tespiti: Segment Düzeyinde Sızıntı

**Sorun:** Ham sinyaller segmentlere bölündükten SONRA rastgele karıştırılıp train/test ayrılıyordu. CWRU sinyalleri periyodik olduğu için, birbirine bitişik/çok benzeyen segmentler train ve test setine dağılabiliyor — model gerçek genelleme yapmıyor, örtüşen bilgiyi "hatırlıyor".

**Çözüm:** Ham sinyal, segmentlemeden ÖNCE train (%80) ve test (%20) olarak ikiye bölündü; segmentleme her parçaya ayrı ayrı uygulandı. Normalizasyon değerleri (min/max) sadece train verisinden hesaplanıp test'e uygulandı (normalizasyon sızıntısını önlemek için).

**Sonuç:** Segment sızıntısı düzeltildikten sonra bile model **yine %100** doğruluk verdi. Bu, başka bir sorunun işareti oldu.

## 5. İkinci Sızıntı Türü: Aynı-Oturum Önyargısı (Same-Session Bias)

**Analiz:** Her sınıfın verisi (`97.mat`, `105.mat`, `118.mat`, `130.mat`) ayrı bir kayıt oturumunda toplanmış. Model, arıza fiziğini değil, her oturuma özgü küçük/ilgisiz farklılıkları (sensör montajı, ortam gürültüsü vb.) öğrenip sınıfı dolaylı olarak doğru tahmin ediyor olabilir — literatürde bilinen bir tuzak.

**Test yöntemi:** Çapraz-yük doğrulaması (cross-load validation):
- **Eğitim:** 0 HP (1797 rpm) verisi
- **Test:** 1 HP (1772 rpm), tamamen farklı bir kayıt oturumundan gelen veri

### Cross-Load Veri Dağılımı

| Sınıf | Train (0 HP) | Test (1 HP) |
|-------|----------------|----------------|
| Normal | 475 (dengeli: kullanılan alt küme) | 944 |
| Inner Race | 235 | 237 |
| Ball | 238 | 236 |
| Outer Race | 237 | 238 |

### Sonuç

```
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00       944
  Inner Race       1.00      1.00      1.00       237
        Ball       1.00      1.00      1.00       236
  Outer Race       1.00      1.00      1.00       238
    accuracy                           1.00      1655
```

**Model, hiç görmediği bir kayıt oturumunda (farklı motor yükü) bile %100 doğruluk verdi.**

## 6. Değerlendirme: Bu Sefer Neden Güvenilir?

Önceki %100 sonuçlarından farklı olarak, bu sonuç aynı-oturum önyargısı ile açıklanamaz çünkü train ve test tamamen farklı fiziksel kayıtlardan geliyor. Bunun yerine sonuç, **CWRU verisetinin bilinen bir özelliğine** işaret ediyor: sınıflar arası fiziksel farklar (genlik, frekans karakteri) o kadar belirgin ki (Normal ±0.2 iken Outer Race ±2.5), literatürde birçok çalışma bu veriset üzerinde %98-100 doğruluk bildiriyor. CWRU, bazı araştırmacılar tarafından "gerçek dünya zorluğunu yeterince temsil etmeyen, nispeten kolay bir benchmark" olarak eleştiriliyor.

**Sonuç olarak:** Model muhtemelen gerçekten genelleme yapıyor (rastgele/yanlış bir kestirme yolu değil), ama bu başarı CWRU'nun nispeten "kolay" doğasından besleniyor — daha zorlu koşullarda (küçük arıza boyutları, düşük sinyal/gürültü oranı, karışık arıza tipleri) aynı performansı vermeyebilir.

---

## 7. Genel Özet — Bugün Öğrenilenler

- Confusion matrix'in, doğruluk yüzdesinin gizlediği sorunları (yanıltıcı doğruluk) ortaya çıkarma önemi
- MSE vs CrossEntropyLoss farkı ve sınıflandırma için uygun loss seçimi
- Model checkpointing (en iyi ağırlıkları kaydetme)
- BatchNorm'un SNN eğitim istikrarına katkısı
- **Veri sızıntısının iki türü:** segment düzeyinde sızıntı (train/test ayrımını segmentlemeden önce yapma gerekliliği) ve aynı-oturum önyargısı (farklı koşullarda toplanmış veriyle çapraz doğrulama gerekliliği)
- Normalizasyon sızıntısından kaçınma (min/max değerlerinin sadece train'den hesaplanması)
- CWRU verisetinin literatürdeki bilinen "kolaylık" özelliği ve bunun sonuçları yorumlarken nasıl dikkate alınması gerektiği

## 8. Sıradaki Adımlar

- [ ] (Opsiyonel) Daha zorlu bir senaryo: farklı arıza boyutlarını (0.007", 0.014", 0.021") karıştırarak test etme
- [ ] (Opsiyonel) Düşük sinyal/gürültü oranı simülasyonu ile modelin dayanıklılığını test etme
- [ ] Donanım kurulumları (Akida kartları, Jetson)
- [ ] Diğer framework'lere geçiş (hoca onayı gerekiyor)

---

*Bu doküman, IMEP kapsamında yürütülen "Nöromorfik Hızlandırıcılarla Kestirimci Bakım" çalışmasının dördüncü bölüm ilerleme notlarını içermektedir.*
