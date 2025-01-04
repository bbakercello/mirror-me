import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os


def audio_to_spectrogram(audio_path, save_path=None):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=22050)

    # Compute the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Save or return the spectrogram
    if save_path:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_spectrogram, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel-frequency spectrogram")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return log_spectrogram


if __name__ == "__main__":
    # Example usage
    audio_file = "example_audio.wav"  # Replace with your file
    output_file = "example_spectrogram.png"
    audio_to_spectrogram(audio_file, output_file)
