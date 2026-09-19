import numpy as np
import scipy.io

# 0 HP (eğitim için) ve 1 HP (test için) verilerini yükleyelim
train_files = {
    0: "datasets/cwru/raw/97.mat",   # Normal, 0 HP
    1: "datasets/cwru/raw/105.mat",  # Inner Race, 0 HP
    2: "datasets/cwru/raw/118.mat",  # Ball, 0 HP
    3: "datasets/cwru/raw/130.mat",  # Outer Race, 0 HP
}

test_files = {
    0: "datasets/cwru/raw/98.mat",   # Normal, 1 HP
    1: "datasets/cwru/raw/106.mat",  # Inner Race, 1 HP
    2: "datasets/cwru/raw/119.mat",  # Ball, 1 HP
    3: "datasets/cwru/raw/131.mat",  # Outer Race, 1 HP
}

class_names = {0: "Normal", 1: "Inner Race", 2: "Ball", 3: "Outer Race"}

def load_de_signal(mat_path):
    mat = scipy.io.loadmat(mat_path)
    for key in mat.keys():
        if key.endswith("_DE_time"):
            return mat[key].flatten()
    raise ValueError(f"DE_time bulunamadı: {mat_path}")

window_size = 1024
stride = 512

def segment_signal(signal, window_size, stride):
    num_windows = (len(signal) - window_size) // stride + 1
    segments = []
    for i in range(num_windows):
        start = i * stride
        segments.append(signal[start : start + window_size])
    return np.array(segments)

# 1) Train verisini (0 HP) hazırla
X_train_list, y_train_list = [], []
for label, path in train_files.items():
    signal = load_de_signal(path)
    segments = segment_signal(signal, window_size, stride)
    X_train_list.append(segments)
    y_train_list.append(np.full(len(segments), label))
    print(f"[TRAIN - 0HP] {class_names[label]}: {len(segments)} segment")

# 2) Test verisini (1 HP) hazırla
X_test_list, y_test_list = [], []
for label, path in test_files.items():
    signal = load_de_signal(path)
    segments = segment_signal(signal, window_size, stride)
    X_test_list.append(segments)
    y_test_list.append(np.full(len(segments), label))
    print(f"[TEST - 1HP] {class_names[label]}: {len(segments)} segment")

# 3) Train setini dengele (en düşük sınıfa göre)
min_train = min(len(x) for x in X_train_list)
X_train_list = [x[:min_train] if len(x) > min_train else x for x in X_train_list]
y_train_list = [y[:min_train] if len(y) > min_train else y for y in y_train_list]

# Rastgele seçim ile dengeleme (daha doğru yöntem)
X_train_balanced, y_train_balanced = [], []
for i, (X_part, y_part) in enumerate(zip(X_train_list, y_train_list)):
    if len(X_part) > min_train:
        idx = np.random.choice(len(X_part), min_train, replace=False)
        X_part = X_part[idx]
        y_part = y_part[idx]
    X_train_balanced.append(X_part)
    y_train_balanced.append(y_part)

X_train = np.concatenate(X_train_balanced, axis=0)
y_train = np.concatenate(y_train_balanced, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# 4) Normalizasyon (sadece train'den öğrenilen min/max ile)
X_min, X_max = X_train.min(), X_train.max()
X_train_norm = (X_train - X_min) / (X_max - X_min)
X_test_norm = np.clip((X_test - X_min) / (X_max - X_min), 0, 1)

print("\nTrain X boyutu:", X_train.shape, "Test X boyutu:", X_test.shape)

np.save("X_train_crossload.npy", X_train_norm)
np.save("y_train_crossload.npy", y_train)
np.save("X_test_crossload.npy", X_test_norm)
np.save("y_test_crossload.npy", y_test)
print("Veri kaydedildi (cross-load).")