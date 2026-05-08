import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import config
from models.cnn_model import BirdCNN
from models.crnn_model import BirdCRNN


def load_model(model_path, model_type, num_classes, device):
    if model_type == "cnn":
        model = BirdCNN(num_classes=num_classes, pretrained=False).to(device)
    else:
        model = BirdCRNN(num_classes=num_classes, pretrained=False).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_top_k(model, image_path, classes, device, k=3):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        top_prob, top_indices = torch.topk(probabilities, k)

        results = []
        for i in range(k):
            results.append({
                "class": classes[top_indices[i]],
                "probability": top_prob[i].item()
            })
        return results, image
    except Exception as e:
        print(f"Hata ({image_path}): {e}")
        return None, None


def visualize_prediction(image, results, save_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Test Spektrogramı")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    labels = [r['class'] for r in results]
    probs = [r['probability'] * 100 for r in results]

    colors = ['#2ecc71', '#3498db', '#95a5a6']
    bars = plt.barh(labels[::-1], probs[::-1], color=colors[::-1])

    plt.xlabel('Olasılık (%)')
    plt.title('En Yüksek 3 Tahmin')
    plt.xlim(0, 100)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f'%{width:.1f}',
            va='center'
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    device = config.DEVICE
    data_dir = config.DATA_DIR
    input_dir = config.TEST_IMAGES_DIR
    output_dir = config.PREDICTIONS_DIR
    model_type = config.MODEL_TYPE
    max_classes = config.MAX_CLASSES

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])[:max_classes]

    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı ({model_path}).")
        return

    model = load_model(model_path, model_type, len(classes), device)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_files:
        print(f"Uyarı: {input_dir} dizininde işlenecek görsel bulunamadı.")
        return

    print(f"{len(image_files)} görsel işleniyor...")

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        results, original_img = predict_top_k(model, img_path, classes, device)

        if results:
            save_name = f"pred_{os.path.splitext(img_name)[0]}.png"
            save_path = os.path.join(output_dir, save_name)
            visualize_prediction(original_img, results, save_path)
            print(f"Kaydedildi: {img_name} -> {save_name}")

    print(f"Sonuçlar '{output_dir}' dizinine kaydedildi.")


if __name__ == "__main__":
    main()
