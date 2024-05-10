# SEM Image Processing and Machine Learning for Ductility and Brittleness Analysis

This repository contains code for analyzing Scanning Electron Microscopy (SEM) images to determine ductility and brittleness percentages using machine learning techniques. The project involves various stages, including image filtering, augmentation, data labeling, model development, and prediction.

## Overview

The project aims to enhance the analysis of SEM images by applying machine learning methods to accurately determine the percentages of ductility and brittleness in materials. The process involves capturing SEM images, preprocessing them, augmenting the dataset, developing a convolutional neural network (CNN) model, and predicting ductility and brittleness percentages.

## Components

### 1. Image Filtering

The `filter_images.py` script implements image filtering techniques to preprocess SEM images. It utilizes OpenCV for image processing tasks such as grayscale conversion, binary thresholding, contour detection, and cropping to remove unwanted parts.

### 2. Augmentation

The `augmentation.py` module provides functions for data augmentation, including zooming, cropping, and skewing. These augmentation techniques help increase the diversity of the training dataset, improving the model's generalization ability.

### 3. Data Labeling and Image Processing

The `data_processing.py` script reads SEM images, calculates ductility and brittleness percentages, and writes the results to an output text file. It employs OpenCV for image processing tasks such as gradient computation and thresholding.

### 4. Machine Learning Model

The `cnn_model.py` script defines a CNN architecture using TensorFlow/Keras for predicting ductility and brittleness percentages. The model comprises convolutional layers, max-pooling layers, and fully connected layers.

### 5. Average Ductility and Brittleness Calculation

The `calculate_statistics.py` script computes the average ductility and brittleness percentages from the output text file generated during data processing.

### 6. Model Predictions

The `predict_model.py` script makes predictions of ductility and brittleness percentages for given SEM images using the trained CNN model. It provides sample predictions for reference.

