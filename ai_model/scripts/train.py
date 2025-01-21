import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import os


def create_model(input_shape):
    """
    Creates and compiles a Convolutional Neural Network model.
    
    Args:
        input_shape (tuple): Shape of the input spectrogram (height, width, channels).

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(input_shape[0] * input_shape[1], activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model




def train_model(data_dir, model_save_path):
    # Load preprocessed spectrograms
    X_train = []
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            X_train.append(np.load(os.path.join(data_dir, file)))
    X_train = np.array(X_train, dtype="float32")

    # Reshape for CNN input
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train[..., np.newaxis]

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(16)

    # Create and train the model
    model = create_model(input_shape)
    model.fit(dataset, epochs=10)

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
