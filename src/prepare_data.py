import os
from tqdm import tqdm
import config
from audio_utils import audio_to_spectrogram, split_audio


def prepare_dataset():
    voice_dir = config.VOICE_DIR
    output_base_dir = config.PROCESSED_DATA_DIR

    bird_species = [
        d for d in os.listdir(voice_dir)
        if os.path.isdir(os.path.join(voice_dir, d)) and not d.startswith('.')
    ]

    if not bird_species:
        print(f"Uyarı: '{voice_dir}' dizininde tür klasörü bulunamadı.")
        return

    print(f"Toplam {len(bird_species)} tür klasörü işleniyor...")

    for species in bird_species:
        species_input_dir = os.path.join(voice_dir, species)
        species_output_dir = os.path.join(output_base_dir, species)
        os.makedirs(species_output_dir, exist_ok=True)

        audio_files = [
            f for f in os.listdir(species_input_dir)
            if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))
        ]

        print(f"Tür: {species} ({len(audio_files)} dosya)")

        for audio_name in tqdm(audio_files):
            audio_path = os.path.join(species_input_dir, audio_name)
            segments, _ = split_audio(audio_path)

            for i in range(segments):
                start_sec = i * config.DURATION
                img = audio_to_spectrogram(audio_path, start_sec=start_sec)

                if img is not None:
                    img_save_name = f"{os.path.splitext(audio_name)[0]}_{i}.png"
                    img_save_path = os.path.join(species_output_dir, img_save_name)
                    img.save(img_save_path)

    print(f"Veri hazırlama tamamlandı: {output_base_dir}")


if __name__ == "__main__":
    prepare_dataset()
