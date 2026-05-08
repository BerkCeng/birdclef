import os
import torch
import config
from data_loader import get_dataloader
from models.cnn_model import BirdCNN
from models.crnn_model import BirdCRNN
from utils import plot_cm, print_metrics, save_classification_report, plot_top_confusions


def evaluate():
    device = config.DEVICE
    print(f"Cihaz: {device}")

    dataloader, num_classes = get_dataloader(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        is_train=False,
        max_classes=config.MAX_CLASSES
    )

    all_classes = sorted([
        d for d in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, d)) and not d.startswith('.')
    ])
    if config.MAX_CLASSES is not None:
        classes = all_classes[:config.MAX_CLASSES]
    else:
        classes = all_classes

    if config.MODEL_TYPE == "cnn":
        model = BirdCNN(num_classes=num_classes, pretrained=False).to(device)
    else:
        model = BirdCRNN(num_classes=num_classes, pretrained=False).to(device)

    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Değerlendirme başlatılıyor...")
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    plot_cm(all_labels, all_preds, classes, save_path=config.CM_PLOT_PATH)
    plot_top_confusions(all_labels, all_preds, classes, save_path=config.CM_SUMMARY_PATH, top_n=30)
    print_metrics(all_labels, all_preds, classes)
    save_classification_report(all_labels, all_preds, classes, save_path=config.REPORT_PATH)


if __name__ == "__main__":
    evaluate()
