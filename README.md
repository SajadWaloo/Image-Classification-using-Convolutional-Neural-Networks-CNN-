# CIFAR-10 Image Classification with Convolutional Neural Networks

This project demonstrates image classification using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. It trains a CNN model to recognize and classify 10 different object classes in the CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. Each image is labeled with one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The dataset is preprocessed by scaling the pixel values between 0 and 1 (float32) to normalize the data before training the model.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python (version 3.6 or later)
- NumPy (```pip install numpy```)
- TensorFlow (```pip install tensorflow```)
- Matplotlib (```pip install matplotlib```)

## Getting Started

To get started with the project, follow these steps:

1. Clone the project repository from GitHub.
2. Install the required dependencies as mentioned in the "Requirements" section.
3. Open a terminal or command prompt and navigate to the project directory.
4. Run the script ```cifar10_classification.py``` using the command: ```python cifar10_classification.py```.
5. The script will load the CIFAR-10 dataset, train the CNN model, plot the training loss and accuracy, and evaluate the model's performance on the test set.
6. The final test accuracy will be displayed in the terminal.

## Model Architecture

The CNN model architecture consists of the following layers:

1. **Conv2D**: 32 filters of size 3x3, ReLU activation function.
2. **MaxPooling2D**: Max pooling with pool size 2x2.
3. **Conv2D**: 64 filters of size 3x3, ReLU activation function.
4. **MaxPooling2D**: Max pooling with pool size 2x2.
5. **Flatten**: Flatten the output from the previous layer.
6. **Dense**: Fully connected layer with 64 units and ReLU activation function.
7. **Dense**: Fully connected layer with 10 units (output layer).

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.

## Results

After training the model for 10 epochs, the training loss and accuracy are plotted using Matplotlib. The final test accuracy of the model is displayed in the terminal.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and adapt the code according to your needs.

If you have any questions or suggestions, please feel free to contact me.

**Author:** Sajad Waloo
---
