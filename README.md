# Rice Image Classification with Convolutional Neural Networks

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) designed to classify different types of rice based on image data. The project includes data preprocessing, model training, evaluation, and saving the trained model for future use.
# Table of Contents

    Overview
    Dataset
    Installation
    Usage
    Model Architecture
    Training
    Evaluation
    Results
    Saving the Model
    License

# Overview

This project uses a CNN to classify images of different rice varieties. The dataset is split into training, validation, and test sets. The model is trained over 5 epochs, and its performance is evaluated using accuracy, precision, and recall metrics.
# Dataset
Link: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset
The dataset used in this project consists of rice images categorized into five classes: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. The images undergo transformations such as resizing and normalization before being used to train the model.

    Classes:
        Arborio
        Basmati
        Ipsala
        Jasmine
        Karacadag

# Installation

To run this project locally, ensure you have the following libraries installed:

bash

pip install torch torchvision matplotlib pandas numpy scikit-learn

Usage

Clone the repository and navigate to the project directory:

bash

git clone https://github.com/your-username/rice-image-classification.git
cd rice-image-classification

Make sure the dataset is correctly placed in the directory specified in the PATH variable in the code. Then, run the script to start training the model:

bash

python train.py

Model Architecture

The CNN model used in this project includes:

    Convolutional layers for feature extraction
    Max-pooling layers for down-sampling
    ReLU activations
    Fully connected layers for classification

Training

The model is trained for 5 epochs, with training loss and validation accuracy being tracked. The training process uses Stochastic Gradient Descent (SGD) as the optimizer and Cross-Entropy as the loss function.
Evaluation

After training, the model is evaluated on the test set. The performance metrics include accuracy, precision, recall, and F1-score. These metrics provide insights into how well the model is classifying each rice variety.
Results

The model achieved a high level of accuracy on the test set, with detailed performance metrics provided in the output.

    Validation Accuracy: 97.25%
    Test Accuracy: 97.15%

Saving the Model

The trained model and optimizer states are saved in a file named rice_CNN.pth, allowing you to reload the model later for further training or inference.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to customize the content, especially the sections on results and dataset paths, according to your specific project details!
