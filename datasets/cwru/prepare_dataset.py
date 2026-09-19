import numpy as np
import scipy.io

mat97 = scipy.io.loadmat("datasets/cwru/raw/97.mat")
mat105 = scipy.io.loadmat("datasets/cwru/raw/105.mat")
mat118 = scipy.io.loadmat("datasets/cwru/raw/118.mat")
mat130 = scipy.io.loadmat("datasets/cwru/raw/130.mat")

signals = {
    0: mat97["X097_DE_time"].flatten(),
    1: mat105["X105_DE_time"].flatten(),
    2: mat118["X118_DE_time"].flatten(),
    3: mat130["X130_DE_time"].flatten(),
}
class_names = {0: "Normal", 1: "Inner Race", 2: "Ball", 3: "Outer Race"}

window_size = 1024
stride = 512
max_train_segments = 188

def segment_signal(signal, window_size, stride):
    num_windows = (len(signal) - window_size) // stride + 1
    segments = []
    for i in range(num_windows):
        start = i * stride
        segments.append(signal[start : start + window_size])
    return np.array(segments)

# YENİ: Her sınıfın ham sinyalini ÖNCE train/test bölgesine ayırıyoruz (segmentlemeden önce!)
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []
for label, signal in signals.items():
    split_point = int(len(signal) * 0.8)
    train_part = signal[:split_point]
    test_part = signal[split_point:]

    train_segments = segment_signal(train_part, window_size, stride)
    test_segments = segment_signal(test_part, window_size, stride)

    # Sadece train'i dengeliyoruz
    if len(train_segments) > max_train_segments:
        indices = np.random.choice(len(train_segments), max_train_segments, replace=False)
        train_segments = train_segments[indices]

    X_train_list.append(train_segments)
    y_train_list.append(np.full(len(train_segments), label))
    X_test_list.append(test_segments)
    y_test_list.append(np.full(len(test_segments), label))

    print(f"{class_names[label]}: {len(train_segments)} train (dengeli), {len(test_segments)} test")

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

X_min, X_max = X_train.min(), X_train.max()
X_train_norm = (X_train - X_min) / (X_max - X_min)
X_test_norm = np.clip((X_test - X_min) / (X_max - X_min), 0, 1)

print("\nTrain X boyutu:", X_train.shape, "Test X boyutu:", X_test.shape)

np.save("X_train_cwru.npy", X_train_norm)
np.save("y_train_cwru.npy", y_train)
np.save("X_test_cwru.npy", X_test_norm)
np.save("y_test_cwru.npy", y_test)
print("Veri kaydedildi.")