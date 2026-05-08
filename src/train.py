import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from data_loader import get_dataloader
from models.cnn_model import BirdCNN
from models.crnn_model import BirdCRNN
from utils import plot_history


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Eğitim')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"%{100 * correct / total:.2f}"})

    return running_loss / len(dataloader), 100 * correct / total


def main():
    device = config.DEVICE
    print(f"Cihaz: {device}")

    dataloader, num_classes = get_dataloader(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        max_classes=config.MAX_CLASSES
    )
    print(f"Sınıf sayısı: {num_classes}")

    if num_classes == 0:
        print(f"Hata: '{config.DATA_DIR}' dizininde sınıf klasörü bulunamadı.")
        return

    if config.MODEL_TYPE == "cnn":
        model = BirdCNN(model_name='resnet34', num_classes=num_classes).to(device)
    else:
        model = BirdCRNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Ortalama Kayıp: {loss:.4f} | Doğruluk: %{acc:.2f}")

        history['train_loss'].append(loss)
        history['train_acc'].append(acc)

        torch.save(model.state_dict(), config.MODEL_PATH)

    plot_history(history, save_path=config.HISTORY_PLOT_PATH)


if __name__ == "__main__":
    main()
