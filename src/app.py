import os
import tempfile
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import config
from gradcam import GradCAM, overlay_heatmap
from models.cnn_model import BirdCNN
from models.crnn_model import BirdCRNN

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(
    page_title="BirdCLEF Spektrogram Sınıflandırma",
    layout="wide"
)


@st.cache_resource
def load_bird_model():
    classes = sorted([
        d for d in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, d)) and not d.startswith('.')
    ])[:config.MAX_CLASSES]

    if config.MODEL_TYPE == "cnn":
        model = BirdCNN(num_classes=len(classes), pretrained=False).to(config.DEVICE)
        target_layer = model.model.layer4[-1]
    else:
        model = BirdCRNN(num_classes=len(classes), pretrained=False).to(config.DEVICE)
        target_layer = model.cnn.layer4[-1]

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    return model, classes, target_layer


def get_prediction(img, model, classes):
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, min(5, len(classes)))
    predictions = [
        {"species": classes[i.item()], "confidence": p.item()}
        for p, i in zip(top_probs, top_idxs)
    ]
    return predictions, input_tensor


st.title("BirdCLEF Spektrogram Sınıflandırma")

try:
    model, classes, target_layer = load_bird_model()
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.metric("Model Mimarisi", config.MODEL_TYPE.upper())
meta_col2.metric("Çalışma Cihazı", str(config.DEVICE))
meta_col3.metric("Sınıf Sayısı", len(classes))

tab_analysis, tab_metrics = st.tabs(["Spektrogram Analizi", "Değerlendirme Metrikleri"])

with tab_analysis:
    img_file = st.file_uploader(
        "Spektrogram görseli seçin (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"]
    )

    if img_file is not None:
        img_in = Image.open(img_file).convert("RGB")
        predictions, input_tensor = get_prediction(img_in, model, classes)

        cam = GradCAM(model, target_layer)
        mask, _ = cam.generate_cam(input_tensor)
        save_p = os.path.join(tempfile.gettempdir(), "gradcam_overlay.png")
        overlay_heatmap(None, mask, save_p, original_img=img_in)

        col_img, col_cam, col_preds = st.columns([1, 1, 1])

        with col_img:
            st.subheader("Girdi Spektrogramı")
            st.image(img_in, use_container_width=True)

        with col_cam:
            st.subheader("Grad-CAM Aktivasyon Haritası")
            st.image(save_p, use_container_width=True)

        with col_preds:
            st.subheader("En Yüksek 5 Olasılık")
            for item in predictions:
                pct = item["confidence"] * 100
                st.write(f"**{item['species']}** — %{pct:.2f}")
                st.progress(item["confidence"])

with tab_metrics:
    history_p = config.HISTORY_PLOT_PATH
    cm_p = config.CM_PLOT_PATH
    cm_summary_p = config.CM_SUMMARY_PATH
    report_p = config.REPORT_PATH

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Eğitim Kayıp ve Doğruluk Grafiği")
        if os.path.exists(history_p):
            st.image(history_p, use_container_width=True)
        else:
            st.info("Eğitim geçmişi grafiği bulunamadı.")

    with col_right:
        st.subheader("En Çok Karıştırılan Sınıflar")
        if os.path.exists(cm_summary_p):
            st.image(cm_summary_p, use_container_width=True)
        elif os.path.exists(cm_p):
            st.image(cm_p, use_container_width=True)
        else:
            st.info("Karmaşıklık matrisi grafiği bulunamadı.")

    st.subheader("Sınıflandırma Raporu")
    if os.path.exists(report_p):
        with open(report_p, "r") as f:
            report_text = f.read()
        st.code(report_text, language="text")
    else:
        st.info("Sınıflandırma raporu dosyası bulunamadı.")
