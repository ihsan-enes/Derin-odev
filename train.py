import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.model import MNIST_MLP
from src.dataset import get_mnist_dataloaders

def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy

def main():
    epochs = 15
    batch_size = 64
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    print(f"Kullanılan Donanım Birimi: {device}")

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir='./data',
        batch_size=batch_size,
        val_split_ratio=0.15
    )

    model = MNIST_MLP(input_size=784, hidden1_size=256, hidden2_size=128, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        epoch_val_loss, epoch_val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Süre: {epoch_duration:.2f}s | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: %{epoch_train_acc:.2f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: %{epoch_val_acc:.2f}")

        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), 'models/best_mlp_model.pth')
            print(f"  -> En iyi model kaydedildi! (Val Acc: %{best_val_accuracy:.2f})")

    total_duration = time.time() - start_total_time
    print(f"\nEğitim Tamamlandı. Toplam Süre: {total_duration:.2f} saniye")

    print("\nEn iyi ağırlıklar yükleniyor ve Nihai Test Kümesi değerlendiriliyor...")
    model.load_state_dict(torch.load('models/best_mlp_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Nihai Test Sonucu -> Loss: {test_loss:.4f} | Accuracy: %{test_acc:.2f}")

    # Grafik Çizimi
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_losses, label='Eğitim Kaybı (Train Loss)', marker='o')
    ax1.plot(epochs_range, val_losses, label='Doğrulama Kaybı (Val Loss)', marker='o')
    ax1.set_title('Kayıp (Loss) Eğrisi')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accuracies, label='Eğitim Başarımı (Train Acc)', marker='o')
    ax2.plot(epochs_range, val_accuracies, label='Doğrulama Başarımı (Val Acc)', marker='o')
    ax2.set_title('Doğruluk (Accuracy) Eğrisi')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Doğruluk (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('reports/training_curves.png')
    plt.close()
    print("Eğitim grafikleri 'reports/training_curves.png' konumuna kaydedildi.")

if __name__ == '__main__':
    main()
