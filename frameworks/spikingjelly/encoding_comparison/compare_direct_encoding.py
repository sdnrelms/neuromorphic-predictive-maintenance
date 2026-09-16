import torch
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron

time_steps = 50
t = np.linspace(0, 4 * np.pi, time_steps)
signal = (np.sin(t) + 1) / 2
signal_tensor = torch.tensor(signal, dtype=torch.float32)

# Eşiği düşürdük: 1.0 yerine 0.4 -- artık sinyalin bir kısmı bunu geçebilecek
lif = neuron.LIFNode(v_threshold=0.4, v_reset=0.0, tau=2.0)

direct_spikes = []
for i in range(time_steps):
    spike = lif(signal_tensor[i].unsqueeze(0))
    direct_spikes.append(spike.item())

print("Toplam spike sayısı:", sum(direct_spikes))
print("Spike'ların geldiği zaman adımları:", np.where(np.array(direct_spikes) == 1)[0])

plt.figure(figsize=(10, 3))
plt.eventplot(np.where(np.array(direct_spikes) == 1)[0], lineoffsets=0, colors='purple')
plt.title("Direct Encoding (v_threshold=0.4)")
plt.xlabel("Zaman Adımı")
plt.savefig("direct_encoding.png")
plt.show()