import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model_path, input_spectrogram):
    """
    Evaluates the model by generating output from an input spectrogram.

    Args:
        model_path (str): Path to the saved model.
        input_spectrogram (numpy.ndarray): Input spectrogram for evaluation.

    Returns:
        numpy.ndarray: Generated output spectrogram.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Normalize input spectrogram
    input_spectrogram = input_spectrogram / np.max(input_spectrogram)

    # Add batch and channel dimensions for the model
    input_data = input_spectrogram[np.newaxis, ..., np.newaxis]  # (1, height, width, channels)

    # Predict output
    output = model.predict(input_data)
    return output.reshape(input_spectrogram.shape)


if __name__ == "__main__":
    # Paths to model and test spectrogram
    model_path = "model/waveform_generator.h5"
    test_spectrogram_path = "data/spectrograms/example_spectrogram.npy"

    # Load test spectrogram
    test_spectrogram = np.load(test_spectrogram_path)

    # Evaluate the model
    generated_output = evaluate_model(model_path, test_spectrogram)

    # Display the output spectrogram
    plt.imshow(generated_output, cmap="hot", interpolation="nearest")
    plt.title("Generated Output")
    plt.colorbar()
    plt.show()
