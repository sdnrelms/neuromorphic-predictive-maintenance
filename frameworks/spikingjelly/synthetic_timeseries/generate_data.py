import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_normal_signal(length=100):
    t = np.linspace(0, 4 * np.pi, length)
    signal = 0.3 * np.sin(t) + 0.5  # düşük genlik, düzenli
    noise = np.random.normal(0, 0.02, length)  # az gürültü
    return signal + noise

def generate_faulty_signal(length=100):
    t = np.linspace(0, 4 * np.pi, length)
    signal = 0.3 * np.sin(t) + 0.5
    # Ani, düzensiz sıçramalar ekleyelim (arıza belirtisi gibi)
    spikes = np.zeros(length)
    spike_positions = np.random.choice(length, size=8, replace=False)
    spikes[spike_positions] = np.random.uniform(0.3, 0.5, size=8)
    noise = np.random.normal(0, 0.05, length)  # daha fazla gürültü
    return signal + spikes + noise

# Veri setini oluşturalım: 200 normal, 200 arızalı örnek
num_samples_per_class = 200
signal_length = 100

normal_signals = np.array([generate_normal_signal(signal_length) for _ in range(num_samples_per_class)])
faulty_signals = np.array([generate_faulty_signal(signal_length) for _ in range(num_samples_per_class)])

# 0-1 aralığına kırpalım (negatif/1'i aşan değerler olabilir)
normal_signals = np.clip(normal_signals, 0, 1)
faulty_signals = np.clip(faulty_signals, 0, 1)

# Etiketler: 0 = normal, 1 = arızalı
X = np.concatenate([normal_signals, faulty_signals], axis=0)
y = np.concatenate([np.zeros(num_samples_per_class), np.ones(num_samples_per_class)])

print("Veri boyutu:", X.shape)  # (400, 100)
print("Etiket boyutu:", y.shape)

# Görselleştirelim: birer örnek
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
axs[0].plot(normal_signals[0], color='green')
axs[0].set_title("Örnek: Normal Sinyal")
axs[1].plot(faulty_signals[0], color='red')
axs[1].set_title("Örnek: Arızalı Sinyal")
plt.tight_layout()
plt.savefig("synthetic_signals.png")
plt.show()

# Veriyi kaydedelim, sonraki adımda kullanacağız
np.save("X_data.npy", X)
np.save("y_data.npy", y)
print("Veri kaydedildi: X_data.npy, y_data.npy")