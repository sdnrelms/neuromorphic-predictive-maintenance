import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import neuron, layer, surrogate, functional

# 1) Veriyi yükle
X = np.load(r"datasets\cwru\X_cwru.npy")  # yol, dosyanızın konumuna göre ayarlanmalı
y = np.load(r"datasets\cwru\y_cwru.npy")

# 2) Karıştır ve train/test ayır
np.random.seed(0)
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train boyutu:", X_train.shape, "Test boyutu:", X_test.shape)

# 3) Dataset sınıfı (sentetik veri denemenizle aynı)
class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SignalDataset(X_train, y_train)
test_dataset = SignalDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

sample_x, sample_y = next(iter(train_loader))
print("Batch X boyutu:", sample_x.shape)
print("Batch y boyutu:", sample_y.shape)

# 4) Model tanımı
class CWRUSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv1d(1, 8, kernel_size=5, padding=2),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool1d(4),   # 1024 -> 256

            layer.Conv1d(8, 16, kernel_size=5, padding=2),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool1d(4),   # 256 -> 64

            layer.Flatten(),
            layer.Linear(16 * 64, 4),   # 4 sınıf: Normal, Inner Race, Ball, Outer Race
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.net(x)

model = CWRUSNN()

# Boyut kontrolü
dummy = torch.rand(1, 1, 1024)
out = model(dummy)
print("Model çıktı boyutu:", out.shape)  # [1, 4] olmalı
functional.reset_net(model)

# 5) Eğitim ve değerlendirme fonksiyonları
def train_one_epoch(model, loader, optimizer, loss_fn, T=10):
    model.train()
    total_loss = 0
    for signals, labels in loader:
        signals = signals.unsqueeze(1)  # [batch, 1024] -> [batch, 1, 1024]
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=4).float()

        optimizer.zero_grad()

        out_sum = 0
        for t in range(T):
            out = model(signals)
            out_sum += out
        out_firing_rate = out_sum / T

        loss = loss_fn(out_firing_rate, labels_onehot)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, T=10):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.unsqueeze(1)
            out_sum = 0
            for t in range(T):
                out = model(signals)
                out_sum += out
            out_firing_rate = out_sum / T
            predicted = out_firing_rate.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            functional.reset_net(model)
    return correct / total

# 6) Eğitim (güncellenmiş: düşük lr + en iyi modeli takip)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

num_epochs = 30
best_accuracy = 0

for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
    accuracy = evaluate(model, test_loader)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Test Doğruluğu: {accuracy*100:.2f}% (En iyi: {best_accuracy*100:.2f}%)")

print(f"\nEğitim tamamlandı. En iyi test doğruluğu: {best_accuracy*100:.2f}%")