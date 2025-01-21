import os
from preprocess import audio_to_spectrogram
from train import train_model
from evaluate import evaluate_model
import numpy as np

# Get the absolute path of the current script
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def preprocess_audio_files(audio_dir, spectrogram_dir):
    if not os.path.exists(spectrogram_dir):
        os.makedirs(spectrogram_dir)

    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, file)
            spectrogram_path = os.path.join(spectrogram_dir, f"{os.path.splitext(file)[0]}.npy")
            print(f"Processing {audio_path}...")
            spectrogram = audio_to_spectrogram(audio_path)
            np.save(spectrogram_path, spectrogram)
            print(f"Saved spectrogram to {spectrogram_path}")

def train_and_evaluate():
    # Define paths relative to the script
    audio_dir = os.path.join(BASE_DIR, "../../examples/audio/dev_audio")
    spectrogram_dir = os.path.join(BASE_DIR, "../data/spectrograms")
    model_path = os.path.join(BASE_DIR, "../model/waveform_generator.h5")
    example_spectrogram_path = os.path.join(spectrogram_dir, "example_spectrogram.npy")

    # Step 1: Preprocess Audio
    preprocess_audio_files(audio_dir, spectrogram_dir)

    # Step 2: Train the Model
    train_model(spectrogram_dir, model_path)

    # Step 3: Evaluate the Model
    test_spectrogram = np.load(example_spectrogram_path)
    output = evaluate_model(model_path, test_spectrogram)

    # Save or visualize the output
    print("Evaluation complete. Output shape:", output.shape)

if __name__ == "__main__":
    train_and_evaluate()
