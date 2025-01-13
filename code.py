import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from google.colab import files, drive
import logging
import os
from google.colab import files
from google.colab import drive

# Constants
IMG_HEIGHT, IMG_WIDTH = 28, 28
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 15
MODEL_FILENAME = "mnist_model.h5"
DRIVE_SAVE_PATH = "/content/drive/MyDrive/models"  # Path in Google Drive
SAVE_PATH = "./models"  # Path to save the model locally

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mount_drive():
    """Mounts Google Drive."""
    try:
        drive.mount('/content/drive')
        logging.info("Google Drive mounted successfully.")
        # Create the directory if it doesn't exist
        os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)
    except Exception as e:
        logging.error(f"Error mounting Google Drive: {e}")

def load_and_preprocess_data():
    """Loads and preprocesses the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize images to range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to (28, 28, 1) to add a channel dimension
    x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    logging.info("Data loaded and preprocessed successfully.")
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
    """Defines and returns a CNN model."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    logging.info("CNN model created successfully.")
    return model

def train_model(model, train_data, val_data):
    """Trains the model and returns the trained model."""
    x_train, y_train = train_data
    x_val, y_val = val_data

    history = model.fit(x_train, y_train, 
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_val, y_val),
                        verbose=1)
    logging.info("Model training completed.")
    return model

def evaluate_model(model, test_data):
    """Evaluates the model on test data."""
    x_test, y_test = test_data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

def save_model(model, filename=MNIST CNN training model):
    """Saves the trained model locally and optionally to Google Drive."""
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model_path = os.path.join(SAVE_PATH, filename)

# # Code to save to Google Drive (commented out)
        # model_drive_path = os.path.join(DRIVE_SAVE_PATH, filename)
        # try:
        #     model.save(model_drive_path)
        #     logging.info(f"Model saved to Google Drive: {model_drive_path}")
        # except Exception as e:
        #     logging.error(f"Error saving model to Drive: {e}")

    try:
        # Save locally
        model.save(model_path)
        logging.info(f"Model saved locally to {model_path}")

        # Download for user
        files.download(model_path)
        logging.info("Model downloaded successfully.")

        # Save to Google Drive if mounted
        if os.path.exists('/content/drive'):
            drive_path = os.path.join(DRIVE_SAVE_PATH, filename)
            model.save(drive_path)
            logging.info(f"Model also saved to Google Drive at {drive_path}")
    except Exception as e:
        logging.error(f"Error saving or downloading model: {e}")

def main():
    """Main function."""
    """Main function."""
    # Mount Drive at the beginning of main (commented out)
    # if not mount_drive():
    #     logging.error("Google Drive mount failed. Exiting.")
    #     return
    try:
        # Mount Google Drive
        mount_drive()

        # Load and preprocess the MNIST dataset
        train_data, test_data = load_and_preprocess_data()

        # Create the CNN model
        model = create_cnn_model()

        # Train the CNN model
        trained_model = train_model(model, train_data, test_data)

        # Evaluate the trained model
        evaluate_model(trained_model, test_data)

        # Save the trained model
        save_model(trained_model)

    except Exception as e:
        logging.exception(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
