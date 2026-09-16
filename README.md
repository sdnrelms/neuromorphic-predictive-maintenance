# Neuromorphic Predictive Maintenance

Nöromorfik hızlandırıcılar (BrainChip Akida) ve klasik GPU (Jetson) üzerinde, kestirimci bakım problemi için Spiking Neural Network (SNN) framework'lerinin benchmark edilmesi.


## Amaç

Kestirimci bakım gibi seyrek-olaylı, zaman-serisi tabanlı bir problemde, nöromorfik donanımın klasik GPU'ya göre benzer doğrulukla daha az enerji harcayıp harcamadığını ölçmek.

## Donanımlar

| Donanım | Durum |
|---|---|
| Raspberry Pi 5 + BrainChip Akida | Planlanıyor |
| Orange Pi 5 + BrainChip Akida | Planlanıyor |
| Jetson Orin Nano/Super | Planlanıyor |

## Framework'ler

| Framework | Durum |
|---|---|
| SpikingJelly | ✅ Başlandı |


## İlerleme Günlüğü

- [Bölüm 1 — SNN Temelleri ve SpikingJelly ile İlk Model](docs/01-snn-fundamentals.md)
- [Bölüm 2 — CNN Mimarisi, Encoding Yöntemleri ve İlk Zaman-Serisi Denemesi](docs/02-cnn-encoding-timeseries.md)

## Klasör Yapısı

```
├── docs/            # Öğrenme günlükleri ve notlar
├── frameworks/       # Framework bazlı kod ve deneyler
├── datasets/          # Veriset notları (ham veri repo'da tutulmaz)
├── hardware/          # Donanım kurulum notları
└── benchmarks/        # Kıyaslama sonuçları
```

## Kurulum

```bash
python -m venv snn_env
snn_env\Scripts\activate   # Windows
pip install torch torchvision spikingjelly matplotlib
```

## Lisans

Bu proje eğitim amaçlıdır.
