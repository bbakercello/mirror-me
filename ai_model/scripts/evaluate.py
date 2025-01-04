import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model_path, input_spectrogram):
    model = tf.keras.models.load_model(model_path)
    output = model.predict(input_spectrogram[np.newaxis, ..., np.newaxis])
    return output.reshape(input_spectrogram.shape)


if __name__ == "__main__":
    model_path = "model/waveform_generator.h5"
    test_spectrogram = np.load("data/spectrograms/example_spectrogram.npy")

    generated_output = evaluate_model(model_path, test_spectrogram)
    plt.imshow(generated_output, cmap="hot", interpolation="nearest")
    plt.title("Generated Output")
    plt.show()
