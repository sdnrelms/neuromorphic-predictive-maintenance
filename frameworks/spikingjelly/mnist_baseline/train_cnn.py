import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, layer, surrogate, functional, encoding

# 1) Veri
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2) Model
class ConvSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(1, 16, kernel_size=3, padding=1),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2),

            layer.Conv2d(16, 32, kernel_size=3, padding=1),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2),

            layer.Flatten(),
            layer.Linear(32 * 7 * 7, 10),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.net(x)

model = ConvSNN()

# 3) Encoder, optimizer, loss
encoder = encoding.PoissonEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

T = 10

# 4) Eğitim ve değerlendirme fonksiyonları (MNIST baseline ile aynı)
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

# 5) Eğitim
num_epochs = 3
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, encoder, T)
    accuracy = evaluate(model, test_loader, encoder, T)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Test Doğruluğu: {accuracy*100:.2f}%")