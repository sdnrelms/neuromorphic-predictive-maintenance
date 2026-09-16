# Bölüm 1: SNN Temelleri ve SpikingJelly ile İlk Model

**Proje:** IMEP - Nöromorfik Hızlandırıcılarla Kestirimci Bakım
**Donanımlar (henüz kurulmadı):** Raspberry Pi 5 + BrainChip Akida, Orange Pi 5 + BrainChip Akida, Jetson Orin Nano/Super
**Framework:** SpikingJelly (ilk seçilen framework)
**Ortam:** Python venv, PyTorch (CPU), SpikingJelly — laptop üzerinde, donanımsız aşama

---

## 1. Kavramsal Temel

### 1.1 SNN (Spiking Neural Network) Nedir?

Klasik yapay sinir ağlarında (ANN) her nöron her an sürekli bir sayı (float aktivasyon) üretir ve bu anında bir sonraki katmana iletilir.

SNN'lerde durum farklıdır:
- Her nöronun bir **zar potansiyeli (membrane potential)** vardır.
- Nöron zamanla gelen "yük" darbelerini (spike'ları) biriktirir.
- Potansiyel bir **eşik (threshold)** değerini geçtiğinde nöron **tek bir spike (0/1)** üretir ve sıfırlanır.

İki temel fark:
- **Zaman boyutu var**: Bilgi "ne zaman ateşlendiği" ile de taşınır (time-steps).
- **Seyreklik (sparsity)**: Nöron sürekli değil, sadece eşiği geçtiğinde ateşler → düşük enerji tüketimi.

### 1.2 Nöromorfik Computing

"Nöromorfik" = beyin-benzeri donanım mimarisi.

Klasik CPU/GPU: senkron saat darbeleriyle sürekli hesaplama yapar, bellek ve işlemci ayrık (von Neumann darboğazı).

Nöromorfik çip (örn. BrainChip Akida):
- **Olay-tabanlı (event-driven)** çalışır — spike gelmediği sürece işlem yapılmaz.
- Bellek ve işlem birimi daha iç içe.
- Sonuç: seyrek/olay-tabanlı verilerde GPU'ya göre çok daha az enerji tüketimi.

### 1.3 LIF (Leaky Integrate-and-Fire) Nöron Modeli

En yaygın kullanılan SNN nöron modeli. Üç davranış:

- **Integrate**: Gelen spike'lardan aldığı yükü zar potansiyeline (`V`) ekler.
- **Leaky**: Yeterince input gelmezse `V` zamanla azalır (sızdırma).
- **Fire**: `V`, eşiği (`threshold`) aştığında spike üretir, `V` sıfırlanır (reset).

Basitleştirilmiş formül (SpikingJelly'nin kullandığı yaklaşım):

```
V(t) = V(t-1) + (input - V(t-1)) / tau
eğer V(t) > threshold:
    spike üret, V(t) = v_reset
```

**Önemli matematiksel gerçek:** Sabit bir girdi verildiğinde `V`, girdi değerine yaklaşır ama **onu geçemez** (asimptotik satürasyon). Yani eşik, girdinin ulaşabileceği maksimum değerden düşük olmalı, yoksa süre ne kadar uzarsa uzasın spike hiç gelmez.

### 1.4 Encoding (Rate / Poisson Coding)

SNN'ler ham sayısal veriyi doğrudan işleyemez, önce **spike dizisine** çevrilmesi gerekir.

**Rate coding mantığı:** Girdi ne kadar büyükse, nöron o kadar sık (yüksek frekansta) spike üretir.

**Poisson encoding:** Girdi değeri, o adımda spike üretme **olasılığını** belirler (garantili değil, olasılıksal) — biyolojik nöronlara daha yakın bir davranış.

Kestirimci bakım bağlamı: Titreşim sensöründen gelen genlik yükseldiğinde (anormallik/arıza belirtisi) → yoğun spike; normal çalışmada → az/hiç spike. Bu da nöromorfik donanımın enerji avantajının temelidir.

### 1.5 Surrogate Gradient (Vekil Gradyan)

Sinir ağı eğitimi (backpropagation) fonksiyonun türevini gerektirir. Ancak spike fonksiyonu (basamak fonksiyonu, 0 veya 1) **türevlenemez**.

Çözüm: İleri yönde (forward) gerçek spike kullanılır, geri yönde (backward) türevlenebilir bir **yaklaşık (vekil) fonksiyon** (örn. Sigmoid, ATan) kullanılır.

```python
neuron.LIFNode(surrogate_function=surrogate.ATan())
```

### 1.6 Neden Kestirimci Bakımda SNN / Nöromorfik Donanım?

- **Veri tipi uyumu:** Kestirimci bakım verisi (titreşim, akustik, sıcaklık) zaten zaman-serisi → SNN'lerin doğal çalışma şekliyle uyumlu.
- **Enerji verimliliği:** Sahada sürekli çalışan bir sensör ünitesi için, sadece anormallik anında "uyanan" bir nöromorfik çip (Akida), sürekli sabit hesaplama yapan bir GPU'ya (Jetson) göre çok daha az enerji harcayabilir.

Projenin temel hipotezi:
> Kestirimci bakım gibi seyrek-olaylı, zaman-serisi tabanlı bir problemde, nöromorfik donanım klasik GPU'ya göre benzer doğrulukla çok daha az enerji harcayabilir mi?

---

## 2. Pratik: SpikingJelly ile İlk Adımlar

### 2.1 Ortam Kurulumu

```bash
python -m venv snn_env
# Windows:
snn_env\Scripts\activate

pip install torch torchvision
pip install spikingjelly
pip install matplotlib
```

Doğrulama:
```python
import torch
import spikingjelly
from spikingjelly.activation_based import neuron

print("Torch version:", torch.__version__)
lif = neuron.LIFNode()
print("LIF nöron objesi oluşturuldu:", lif)
```

### 2.2 Tek Bir LIF Nöronunu Gözlemleme

```python
import torch
from spikingjelly.activation_based import neuron

lif = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=2.0)
input_current = torch.tensor([0.3])

for t in range(10):
    spike = lif(input_current)
    v = lif.v.item()
    print(f"{t}: v={v:.3f}, spike={spike.item()}")
```

**Gözlem:** Girdi=0.3 ile potansiyel 0.3'te satüre oldu, hiç spike gelmedi (eşik=1.0 girdiden yüksek olduğu için).

Girdi=1.5 ile denendiğinde: potansiyel eşiği geçip periyodik olarak ateşledi (yükleniyor → ateşliyor → sıfırlanıyor).

### 2.3 Girdi Büyüklüğü — Spike Sayısı İlişkisi (Rate Coding)

```python
def count_spikes(input_value, tau=2.0, steps=50):
    lif = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau)
    input_current = torch.tensor([input_value])
    spike_count = 0
    for t in range(steps):
        spike = lif(input_current)
        spike_count += spike.item()
    return spike_count

for val in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]:
    print(f"Girdi={val:.1f}  ->  50 adımda {int(count_spikes(val))} spike")
```

**Sonuç:**
```
Girdi=0.3  ->  0 spike
Girdi=0.6  ->  0 spike
Girdi=1.0  ->  2 spike
Girdi=1.5  ->  25 spike
Girdi=2.0  ->  50 spike
Girdi=3.0  ->  50 spike
```

Girdi büyüklüğü arttıkça spike sıklığı artıyor → **rate coding**'in temel davranışı.

### 2.4 Gerçek Bir Sinyali Spike Dizisine Çevirme (Poisson Encoding)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import encoding

time_steps = 100
t = np.linspace(0, 4 * np.pi, time_steps)
signal = (np.sin(t) + 1) / 2  # 0-1 aralığına normalize edilmiş sinyal
signal_tensor = torch.tensor(signal, dtype=torch.float32)

encoder = encoding.PoissonEncoder()

spikes = []
for i in range(time_steps):
    spike = encoder(signal_tensor[i].unsqueeze(0))
    spikes.append(spike.item())

spikes = np.array(spikes)

fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axs[0].plot(t, signal, color='blue')
axs[0].set_title("Orijinal Sinyal (örn. normalize edilmiş titreşim genliği)")
axs[1].eventplot(np.where(spikes == 1)[0], lineoffsets=0, colors='red')
axs[1].set_title("Spike Dizisi (Poisson Encoding)")
plt.tight_layout()
plt.savefig("spike_encoding.png")
plt.show()
```

**Gözlem:** Sinyalin tepe noktalarında (genlik yüksek) spike'lar yoğunlaşıyor, çukurlarda (genlik düşük) spike sayısı azalıyor/kayboluyor. Bu, kestirimci bakımda "anormal titreşim → yoğun spike, normal çalışma → sessizlik" mantığının küçük ölçekli bir modeli.

---

## 3. MNIST ile İlk Öğrenen SNN

### 3.1 Neden MNIST?

Kestirimci bakım verisine geçmeden önce, framework'ün temel iş akışını (veri → encoding → model → eğitim → değerlendirme) bilinen, basit bir problem üzerinde öğrenmek için standart bir "merhaba dünya" örneği.

### 3.2 Model Mimarisi

```python
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate

class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(128, 10),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.net(x)
```

Basit bir tam-bağlantılı (fully connected) ağ: 784 piksel → 128 gizli nöron → 10 çıktı sınıfı (rakam 0-9), her katmandan sonra LIF spiking katmanı.

### 3.3 Debugging Süreci — Neden İlk Denemeler Spike Üretmedi?

Bu süreç, LIF'in matematiksel davranışını pratikte anlamak için önemli oldu:

1. **Eğitilmemiş model ile tek adım deneme:** Rastgele başlangıç ağırlıkları çok küçük değerler ürettiği için (`fc1` çıktısı -0.83 ile 0.80 arası), eşik (1.0) hiç aşılamadı → 0 spike.

2. **Aynı girdiyi 10 adım tekrarlama:** Girdinin maksimum değeri (~0.71-0.87) yine eşiğin (1.0) altında kaldığı için, LIF'in satürasyon davranışı gereği (potansiyel girdiyi asla geçemez) süre uzatmak yetmedi → 0 spike.

3. **Eşiği düşürme (v_threshold=0.5):** Eşik artık girdinin maksimum değerinden düşük olduğu için, bazı nöronlar periyodik olarak ateşlemeye başladı → sorun doğrulandı ve çözüldü.

**Çıkarım:** LIF nöronunda spike üretimi için *süre* değil, *girdi büyüklüğünün eşiğe göre yeterliliği* belirleyicidir. Gerçek eğitimde bu sorun, ağırlıkların öğrenme yoluyla otomatik ayarlanmasıyla ve/veya uygun encoding stratejisiyle doğal olarak çözülür — eşiği elle ayarlamaya gerek kalmaz.

### 3.4 Tam Eğitim Döngüsü

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, layer, surrogate, functional, encoding

# Veri
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(128, 10),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.net(x)

model = SimpleSNN()

# Encoder, optimizer, loss
encoder = encoding.PoissonEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

T = 10  # zaman adımı sayısı

def train_one_epoch(model, loader, optimizer, loss_fn, encoder, T):
    model.train()
    total_loss = 0
    for images, labels in loader:
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
        optimizer.zero_grad()
        out_sum = 0
        for t in range(T):
            encoded_img = encoder(images)
            out = model(encoded_img)
            out_sum += out
        out_firing_rate = out_sum / T
        loss = loss_fn(out_firing_rate, labels_onehot)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, encoder, T):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            out_sum = 0
            for t in range(T):
                encoded_img = encoder(images)
                out = model(encoded_img)
                out_sum += out
            out_firing_rate = out_sum / T
            predicted = out_firing_rate.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            functional.reset_net(model)
    return correct / total

num_epochs = 3
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, encoder, T)
    accuracy = evaluate(model, test_loader, encoder, T)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Test Doğruluğu: {accuracy*100:.2f}%")
```

**Önemli detaylar:**
- `labels_onehot`: 10 boyutlu çıktıyı gerçek etiketle (örn. rakam "5") karşılaştırabilmek için one-hot forma çevirme.
- `functional.reset_net(model)`: Her batch sonrası nöronların hafızasını (membran potansiyelini) sıfırlama — yoksa önceki görüntünün etkisi bir sonrakine karışır.
- `argmax(dim=1)`: En yüksek ateşleme oranına sahip sınıf, modelin tahmini olarak alınır.

### 3.5 Sonuçlar

```
Epoch 1/3 - Loss: 0.0207 - Test Doğruluğu: 93.82%
Epoch 2/3 - Loss: 0.0093 - Test Doğruluğu: 95.68%
Epoch 3/3 - Loss: 0.0071 - Test Doğruluğu: 96.39%
```

Loss sürekli düşüyor, doğruluk sürekli artıyor → sağlıklı bir eğitim. Basit bir tam-bağlantılı SNN, 3 epoch sonunda **%96.39** test doğruluğuna ulaştı.

---

## 4. Genel Özet — Bugün Öğrenilenler

- SNN'in klasik ANN'den farkı (zaman boyutu, spike, sparsity)
- Nöromorfik donanımın enerji verimliliği mantığı
- LIF nöron modeli (integrate-leak-fire) ve matematiksel satürasyon davranışı
- Rate / Poisson encoding ile sayısal veriyi spike dizisine çevirme
- Surrogate gradient ile spike fonksiyonunun türevlenemez probleminin çözümü
- SpikingJelly ile uçtan uca bir SNN modeli kurma, eğitme ve değerlendirme
- Debugging: eşik/girdi ilişkisi, satürasyon, eğitilmemiş model davranışı

## 5. Sıradaki Adımlar

- [ ] Framework planlamasının netleşmesi (SpikingJelly ile devam, sonraki 4 framework: muhtemelen snnTorch, Norse, BindsNET, Lava veya benzerleri)
- [ ] Kestirimci bakım veriseti seçimi (CWRU bearing dataset, NASA CMAPSS gibi adaylar)
- [ ] Zaman-serisi sensör verisine uygun encoding stratejisi geliştirme
- [ ] Donanımların (Akida kartları, Jetson) kurulumu ve SDK entegrasyonu (BrainChip MetaTF/CNN2SNN)
- [ ] Benchmark metriklerinin netleştirilmesi: doğruluk, gecikme (latency), güç tüketimi (Joule/inference)
- [ ] 3 donanım × 5 framework × seçilen veriset(ler)i üzerinde kıyaslama tablosu

---

*Bu doküman, IMEP kapsamında yürütülen "Nöromorfik Hızlandırıcılarla Kestirimci Bakım" çalışmasının ilk bölüm ilerleme notlarını içermektedir.*
