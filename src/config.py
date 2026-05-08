import os
import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "data", "test_images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "batch_predictions")
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, "gradcam")
VOICE_DIR = os.path.join(BASE_DIR, "bird_voice")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "bird_voice_image")

for folder in [
    MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, PREDICTIONS_DIR,
    TEST_IMAGES_DIR, GRADCAM_DIR, VOICE_DIR, PROCESSED_DATA_DIR, DATA_DIR
]:
    os.makedirs(folder, exist_ok=True)

MODEL_TYPE = "crnn"
MAX_CLASSES = None
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

SAMPLE_RATE = 32000
DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = 16000

MODEL_NAME = f"kus_modeli_{MODEL_TYPE}_son.pth"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
HISTORY_PLOT_PATH = os.path.join(LOGS_DIR, f"egitim_gecmisi_{MODEL_TYPE}.png")
CM_PLOT_PATH = os.path.join(LOGS_DIR, f"karmasiklik_matrisi_{MODEL_TYPE}.png")
REPORT_PATH = os.path.join(LOGS_DIR, f"siniflandirma_raporu_{MODEL_TYPE}.txt")
CM_SUMMARY_PATH = os.path.join(LOGS_DIR, f"en_cok_karisan_siniflar_{MODEL_TYPE}.png")
