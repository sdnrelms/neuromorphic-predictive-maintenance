import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import neuron, layer, surrogate, functional, encoding

# 1) Kaydettiğimiz veriyi yükleyelim
X = np.load("X_data.npy")  # (400, 100)
y = np.load("y_data.npy")  # (400,)

# 2) Karıştırıp train/test ayıralım (%80 train, %20 test)
np.random.seed(0)
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train boyutu:", X_train.shape, "Test boyutu:", X_test.shape)

# 3) PyTorch Dataset sınıfı oluşturalım
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

print("Bir batch örneği:")
sample_x, sample_y = next(iter(train_loader))
print("X boyutu:", sample_x.shape)  # [16, 100]
print("y boyutu:", sample_y.shape)  # [16]

# 4) Model tanımı
class TimeSeriesSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv1d(1, 8, kernel_size=5, padding=2),   # 1 kanal (ham sinyal) -> 8 kanal
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool1d(2),                              # 100 -> 50

            layer.Conv1d(8, 16, kernel_size=5, padding=2),   # 8 -> 16 kanal
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool1d(2),                              # 50 -> 25

            layer.Flatten(),
            layer.Linear(16 * 25, 2),                        # 2 sınıf: normal / arızalı
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.net(x)

model = TimeSeriesSNN()

# Model boyutlarını kontrol edelim (dummy input ile)
dummy = torch.rand(1, 1, 100)  # [batch, kanal, zaman_adımı]
out = model(dummy)
print("Model çıktı boyutu:", out.shape)  # [1, 2] olmalı
functional.reset_net(model)


# 5) Eğitim ve değerlendirme fonksiyonları
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for signals, labels in loader:
        # signals boyutu: [batch, 100] -> Conv1d için [batch, 1, 100] olmalı (kanal boyutu ekliyoruz)
        signals = signals.unsqueeze(1)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).float()

        optimizer.zero_grad()

        # Burada T=100 (sinyalin kendi uzunluğu) zaman adımı olarak kullanılıyor
        # Her adımda sinyalin İLGİLİ ANI (tek bir sütun) modele veriliyor
        out_sum = 0
        T = signals.shape[2]  # 100
        for t in range(T):
            step_input = signals[:, :, t].unsqueeze(2)  # [batch, 1, 1] - o anki değer
            # Conv1d en az kernel_size uzunluğunda girdi ister, bu yüzden basit modelde
            # tüm sinyali bir kerede vermek daha pratik olacak (aşağıdaki alternatif yaklaşım)
            pass

        # ALTERNATİF VE DAHA PRATİK YAKLAŞIM:
        # Sinyalin tamamını tek seferde modele veriyoruz (Conv1d zaten zaman eksenini işliyor),
        # ama SNN mantığını korumak için modele T=10 kere aynı sinyali besleyip
        # nöronların zamanla "toplanmasını" sağlıyoruz (MNIST'teki gibi)
        out_sum = 0
        T = 10
        for t in range(T):
            out = model(signals)  # tüm sinyali Conv1d'ye veriyoruz
            out_sum += out
        out_firing_rate = out_sum / T

        loss = loss_fn(out_firing_rate, labels_onehot)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.unsqueeze(1)
            out_sum = 0
            T = 10
            for t in range(T):
                out = model(signals)
                out_sum += out
            out_firing_rate = out_sum / T
            predicted = out_firing_rate.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            functional.reset_net(model)
    return correct / total

# 6) Eğitim
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

num_epochs = 15
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
    accuracy = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Test Doğruluğu: {accuracy*100:.2f}%")