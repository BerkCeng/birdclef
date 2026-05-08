import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
import config
from audio_utils import audio_to_spectrogram, split_audio
from models.cnn_model import BirdCNN
from models.crnn_model import BirdCRNN


def load_model(classes_count):
    device = config.DEVICE
    model_type = config.MODEL_TYPE

    if model_type == "cnn":
        model = BirdCNN(num_classes=classes_count, pretrained=False).to(device)
    else:
        model = BirdCRNN(num_classes=classes_count, pretrained=False).to(device)

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {config.MODEL_PATH}")

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def main(audio_path):
    classes = sorted([
        d for d in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, d)) and not d.startswith('.')
    ])[:config.MAX_CLASSES]

    model, device = load_model(len(classes))

    segments, total_duration = split_audio(audio_path)
    print(f"Dosya: {os.path.basename(audio_path)} | Süre: {total_duration:.2f}s | Segment: {segments}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = []

    with torch.no_grad():
        for i in range(segments):
            start_sec = i * config.DURATION
            img = audio_to_spectrogram(audio_path, start_sec=start_sec)

            if img is not None:
                input_tensor = transform(img).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

                prob, idx = torch.max(probabilities, dim=0)
                confidence = prob.item()
                bird_name = classes[idx.item()]

                time_label = f"{start_sec}-{start_sec + config.DURATION}s"
                print(f"[{time_label}] {bird_name} (%{confidence * 100:.1f})")

                results.append({
                    "time": time_label,
                    "bird": bird_name,
                    "confidence": confidence
                })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python3 src/audio_inference.py <ses_dosyasi_yolu>")
        sys.exit(1)
    main(sys.argv[1])
