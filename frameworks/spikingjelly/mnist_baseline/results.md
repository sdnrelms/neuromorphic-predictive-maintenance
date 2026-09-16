# MNIST Baseline - SpikingJelly
 
## Model 1: Tam-Bağlantılı (Linear) SNN
 
Mimari: 784 → 128 → 10, LIF nöron, Poisson encoding, T=10 zaman adımı.
 
| Epoch | Loss   | Test Doğruluğu |
|-------|--------|-----------------|
| 1     | 0.0207 | 93.82%          |
| 2     | 0.0093 | 95.68%          |
| 3     | 0.0071 | 96.39%          |
 
## Model 2: Evrişimli (Convolutional) SNN
 
Mimari: Conv2d(1→16) → LIF → MaxPool → Conv2d(16→32) → LIF → MaxPool → Linear(1568→10) → LIF, Poisson encoding, T=10 zaman adımı.
 
| Epoch | Loss   | Test Doğruluğu |
|-------|--------|-----------------|
| 1     | 0.0274 | 96.91%          |
| 2     | 0.0067 | 97.83%          |
| 3     | 0.0051 | 98.15%          |
 
## Karşılaştırma
 
| Model | Epoch 3 Doğruluk |
|-------|-------------------|
| Linear SNN | 96.39% |
| Conv SNN | **98.15%** |
 
**Yorum:** CNN mimarisi, görüntüdeki yerel örüntüleri (kenar, köşe vb.) yakalayabildiği için Linear modele göre daha yüksek doğruluk sağladı. Bu sonuç, kestirimci bakım verisinde de (zaman serisi sinyallerindeki yerel örüntüleri yakalamak için) Conv1d tabanlı bir mimariye yönelmenin mantıklı olacağını gösteriyor.