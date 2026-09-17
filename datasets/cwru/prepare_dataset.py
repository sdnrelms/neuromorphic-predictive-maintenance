import numpy as np
import scipy.io

# 1) Ham verileri yükleyelim
mat97 = scipy.io.loadmat("datasets/cwru/raw/97.mat")
mat105 = scipy.io.loadmat("datasets/cwru/raw/105.mat")
mat118 = scipy.io.loadmat("datasets/cwru/raw/118.mat")
mat130 = scipy.io.loadmat("datasets/cwru/raw/130.mat")

signals = {
    0: mat97["X097_DE_time"].flatten(),   # 0 = Normal
    1: mat105["X105_DE_time"].flatten(),  # 1 = Inner Race
    2: mat118["X118_DE_time"].flatten(),  # 2 = Ball
    3: mat130["X130_DE_time"].flatten(),  # 3 = Outer Race
}

class_names = {0: "Normal", 1: "Inner Race", 2: "Ball", 3: "Outer Race"}

# 2) Pencereleme fonksiyonu
window_size = 1024
stride = 512


def segment_signal(signal, window_size, stride):
    num_windows = (len(signal) - window_size) // stride + 1
    segments = []
    for i in range(num_windows):
        start = i * stride
        segment = signal[start : start + window_size]
        segments.append(segment)
    return np.array(segments)

# 3) Her sınıf için segmentleri çıkaralım
max_segments_per_class = 235

X_list = []
y_list = []

# for label, signal in signals.items():
#     segments = segment_signal(signal, window_size)
#     # Eğer bu sınıfta max_segments_per_class'tan fazla segment varsa, rastgele o kadarını seçelim
#     if len(segments) > max_segments_per_class:
#         indices = np.random.choice(len(segments), max_segments_per_class, replace=False)
#         segments = segments[indices]
#     X_list.append(segments)
#     y_list.append(np.full(len(segments), label))
#     print(f"{class_names[label]}: {len(segments)} segment kullanıldı")

for label, signal in signals.items():
    segments = segment_signal(signal, window_size, stride)
    if len(segments) > max_segments_per_class:
        indices = np.random.choice(len(segments), max_segments_per_class, replace=False)
        segments = segments[indices]
    print(f"{class_names[label]}: {len(segments)} segment kullanıldı")
    X_list.append(segments)
    y_list.append(np.full(len(segments), label))

# 4) Hepsini birleştirelim
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print("\nToplam X boyutu:", X.shape)
print("Toplam y boyutu:", y.shape)

# 5) Normalize edelim (0-1 aralığına) - LIF/encoding için gerekli
X_min = X.min()
X_max = X.max()
X_normalized = (X - X_min) / (X_max - X_min)

print("Normalize sonrası min/max:", X_normalized.min(), X_normalized.max())

# 6) Kaydedelim
np.save("X_cwru.npy", X_normalized)
np.save("y_cwru.npy", y)
print("\nVeri kaydedildi: X_cwru.npy, y_cwru.npy")