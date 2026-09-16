import torch
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import encoding

# Aynı sinyali kullanacağız (titreşim sensörü benzetmesi)
time_steps = 50
t = np.linspace(0, 4 * np.pi, time_steps)
signal = (np.sin(t) + 1) / 2  # 0-1 arası normalize
signal_tensor = torch.tensor(signal, dtype=torch.float32)

# --- 1) Poisson Encoding ---
poisson_encoder = encoding.PoissonEncoder()
poisson_spikes = []
for i in range(time_steps):
    spike = poisson_encoder(signal_tensor[i].unsqueeze(0))
    poisson_spikes.append(spike.item())

# --- 2) Latency Encoding ---
# LatencyEncoder, TÜM zaman serisini tek seferde alır ve her T adımında
# hangi nöronların ateşleyeceğini önceden hesaplar
latency_encoder = encoding.LatencyEncoder(T=time_steps)
# Latency encoder tek bir değer setini bekler (batch, features) formatında
single_values = signal_tensor.unsqueeze(0)  # [1, 50] -> her biri ayrı bir "özellik" gibi davranacak
latency_spikes_all = []
for t_step in range(time_steps):
    spike = latency_encoder(single_values)
    latency_spikes_all.append(spike.squeeze().numpy())
latency_spikes_all = np.array(latency_spikes_all)  # [time_steps, 50]

# --- Görselleştirme ---
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axs[0].plot(t, signal, color='blue')
axs[0].set_title("Orijinal Sinyal")

axs[1].eventplot(np.where(np.array(poisson_spikes) == 1)[0], lineoffsets=0, colors='red')
axs[1].set_title("Poisson (Rate) Encoding")
axs[1].set_yticks([])

# Latency encoding: her zaman noktası kendi latency paternini üretiyor,
# basitlik için sadece ilk birkaç değerin latency'sini gösterelim
sample_indices = [5, 25, 45]  # düşük, orta, yüksek genlik noktaları
for idx in sample_indices:
    spike_times = np.where(latency_spikes_all[:, idx] == 1)[0]
    axs[2].eventplot(spike_times, lineoffsets=idx/10, colors='green')
axs[2].set_title(f"Latency Encoding (örnek noktalar: genlik={signal[5]:.2f}, {signal[25]:.2f}, {signal[45]:.2f})")
axs[2].set_xlabel("Zaman Adımı")

plt.tight_layout()
plt.savefig("encoding_comparison.png")
plt.show()

print("Poisson toplam spike sayısı:", sum(poisson_spikes))
print("Düşük genlik (", signal[5], ") ilk spike zamanı:", np.where(latency_spikes_all[:,5]==1)[0][:1])
print("Yüksek genlik (", signal[45], ") ilk spike zamanı:", np.where(latency_spikes_all[:,45]==1)[0][:1])