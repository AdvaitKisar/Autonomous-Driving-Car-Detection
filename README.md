# Project Title: Autonomous Driving Car Detection Using YOLO and Non-Max Suppression

## Overview
This project implements a Convolutional Neural Network (CNN) using the YOLO (You Only Look Once) algorithm combined with Non-Max Suppression (NMS) to detect cars in images captured from the front cameras of various vehicles. The dataset utilized for this project is sourced from drive.ai, which provides a comprehensive collection of annotated images for training and evaluation. This project was developed as part of the Coursera Deep Learning Specialization, specifically within the Convolutional Neural Networks course.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
The primary goal of this project is to accurately detect cars in images using advanced deep learning techniques. YOLO is a state-of-the-art, real-time object detection system that processes images in a single pass, making it significantly faster than traditional methods. Non-Max Suppression is employed to refine the model's predictions by eliminating redundant bounding boxes, thereby enhancing detection accuracy.

## Installation
To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/AdvaitKisar/Autonomous-Driving-Car-Detection.git
   cd Autonomous-Driving-Car-Detection
   ```

2. Install the required libraries manually. You will need the following Python packages:
   - TensorFlow or PyTorch (depending on your implementation)
   - OpenCV
   - NumPy
   - Matplotlib

   You can install these using pip:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib
   ```

## Usage
To run the car detection model, open the Jupyter Notebook provided in this repository. The notebook contains all the necessary code to load the dataset, train the model, and perform car detection on sample images.

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the notebook file (e.g., `Autonomous-Driving-Car-Detection.ipynb`) and follow the instructions within to execute each cell.

## Algorithm Details
### YOLO Algorithm
YOLO divides the input image into a grid and predicts bounding boxes and class probabilities for each grid cell. The algorithm processes images in real-time, making it suitable for applications requiring immediate feedback.

### Non-Max Suppression (NMS)
NMS is a post-processing technique that eliminates overlapping bounding boxes based on their confidence scores. It works by retaining only the box with the highest confidence score among those that overlap significantly, thus reducing false positives and ensuring that each detected object is represented by a single bounding box.

## Dataset
The dataset used in this project is provided by drive.ai, containing numerous images of cars captured from various angles and conditions. This diverse dataset helps in training a robust model capable of generalizing well to new images.

## Results
The implementation demonstrates high accuracy in detecting cars across different scenarios. The use of NMS significantly improves the model's performance by refining the output bounding boxes, leading to cleaner and more precise detections.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.
