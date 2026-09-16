# Encoding Yöntemleri Karşılaştırması - SpikingJelly

Aynı sinyal (0-1 arası normalize edilmiş sinüs dalgası, 50 zaman adımı) üç farklı encoding yöntemiyle spike dizisine çevrildi.

## 1) Poisson (Rate) Encoding

`encoding.PoissonEncoder()` kullanıldı. Sinyal değeri, o adımda spike üretme **olasılığını** belirler.

- Sinyalin yüksek olduğu bölgelerde (tepe noktalar) spike yoğunluğu arttı.
- Sinyalin düşük olduğu bölgelerde (çukurlar) spike sayısı azaldı/kayboldu.
- **Olasılıksal**: aynı sinyal tekrar verilse, farklı bir spike paterni çıkabilir.

## 2) Latency Encoding

`encoding.LatencyEncoder(T=time_steps)` kullanıldı. Bilgi, **ilk spike'ın zamanında** taşınır.

Örnek 3 nokta:

| Genlik | İlk Spike Zamanı |
|--------|-------------------|
| 0.98 (yüksek) | ~1 (çok erken) |
| 0.56 (orta) | ~21 (orta) |
| 0.07 (düşük) | ~45 (çok geç) |

**Genlik yüksekse spike erken gelir, düşükse geç gelir.** Çok az sayıda spike ile (tek spike bile yeterli) bilgi taşıyabildiği için enerji açısından en verimli yöntem.

## 3) Direct Encoding

Ham sinyal, herhangi bir encoder kullanılmadan doğrudan bir LIF nörona verildi (`v_threshold=0.4`).

- Toplam 19 spike üretildi, sinyalin iki tepe noktasına denk gelen iki kümede yoğunlaştı.
- **Deterministik**: aynı sinyal tekrar verilirse birebir aynı spike paterni çıkar (Poisson'dan farkı budur).
- **Önemli not:** İlk denemede eşik (1.0) sinyalin maksimum değerinden yüksek olduğu için hiç spike üretilmedi (LIF'in satürasyon davranışı — potansiyel girdiyi asla geçemez). Eşik 0.4'e düşürülünce spike üretimi gözlemlendi. Bu, direct encoding kullanırken veri ölçeği ile eşik değerinin birbirine göre dikkatli ayarlanması gerektiğini gösteriyor.

## Genel Karşılaştırma

| Yöntem | Bilgi Taşıma Şekli | Enerji Verimliliği | Kullanım Senaryosu |
|--------|----------------------|----------------------|----------------------|
| Poisson (Rate) | Spike sıklığı | Orta-yüksek | Gürültüye dayanıklı, genel amaçlı |
| Latency | İlk spike zamanı | Çok düşük | Hızlı tepki / enerji kritik senaryolar |
| Direct | LIF'in kendi dinamiği (deterministik) | Ayarlanabilir (eşiğe bağlı) | Basit prototipleme, manuel eşik ayarı gerekir |

## Kestirimci Bakım Projesi İçin Çıkarım

Latency encoding, enerji verimliliği açısından en güçlü aday olarak öne çıkıyor — "anormallik ne kadar şiddetliyse o kadar hızlı alarm ver" mantığıyla doğal olarak örtüşüyor. Ancak gürültüye karşı dayanıklılığı gerçek sensör verisi üzerinde test edilmeli. Gerçek veriye geçildiğinde üç yöntem de denenip karşılaştırılmalı.