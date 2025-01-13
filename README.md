# MNIST-CNN-Classifier-
CNN Model Classifier on MNIST Data
# MNIST CNN Training

This repository contains code to train a Convolutional Neural Network (CNN) on the MNIST dataset using TensorFlow and Keras. The program includes functionality for saving the trained model locally, downloading it directly, and optionally saving it to Google Drive.

---

## Features

- **Dataset**: Utilizes the MNIST dataset of handwritten digits.
- **CNN Architecture**: Implements a CNN with convolutional, pooling, and dense layers.
- **Training**: Supports training with customizable epochs and batch size.
- **Evaluation**: Evaluates the trained model's accuracy and loss on the test set.
- **Model Persistence**:
  - Saves the model locally.
  - Optionally saves the model to Google Drive if mounted.
  - Provides a direct download option for Colab users.

---

## Requirements

The code is designed to run in a Google Colab environment but can be adapted for local execution. Ensure the following libraries are installed:

- TensorFlow
- Google Colab
- NumPy
- Logging

To install the required libraries:
```bash
pip install tensorflow
```

---

## File Structure

- `mnist_cnn_training.py`: The main Python script containing the code.
- `README.md`: This documentation file.

---

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/mnist-cnn-training.git
   cd mnist-cnn-training
   ```

2. **Run the Code**:
   - Open the script in Google Colab or run it locally if using Colab-specific functions like `files.download()`.
   
3. **Save the Model**:
   - Mount Google Drive for persistent storage.
   - Download the trained model directly for use in other applications.

---

## Example Workflow

1. **Load and Preprocess Data**:
   - Loads MNIST dataset.
   - Normalizes pixel values and reshapes images.

2. **Define the CNN Model**:
   - Includes convolutional layers, pooling layers, and dense layers for classification.

3. **Train the Model**:
   - Configured for 15 epochs and a batch size of 128 (customizable).

4. **Evaluate and Save the Model**:
   - Evaluates accuracy and loss on the test dataset.
   - Saves the model locally and optionally to Google Drive.

---

## Output

- **Logs**: Detailed logs for each step (data loading, training, evaluation, saving).
- **Trained Model**: Saved in `.h5` format for portability.

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

Seyed Parsa Behjat Tabrizi

---

## Acknowledgements

- TensorFlow and Keras documentation.
- Google Colab for seamless execution.
