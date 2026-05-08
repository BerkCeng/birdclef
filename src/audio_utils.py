import librosa
import numpy as np
from PIL import Image
import config


def audio_to_spectrogram(audio_path, start_sec=0, duration=config.DURATION):
    try:
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, offset=start_sec, duration=duration)

        target_len = config.SAMPLE_RATE * duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))

        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            fmin=config.FMIN,
            fmax=config.FMAX
        )

        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        melspec_norm = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)
        melspec_img = (melspec_norm * 255).astype(np.uint8)

        img = Image.fromarray(melspec_img).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Ses işleme hatası ({audio_path}): {e}")
        return None


def split_audio(audio_path):
    duration = librosa.get_duration(path=audio_path)
    segments = int(np.ceil(duration / config.DURATION))
    return segments, duration
