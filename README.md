---
# Indian Sign Language Recognition and Landmark Extraction
---
This project focuses on real-time recognition of Indian Sign Language and the extraction of hand landmarks from videos using OpenCV, MediaPipe, and a trained deep learning model. The goal is to capture hand gestures and recognize them in real-time or from pre-recorded videos, extracting keypoints for further analysis or visualization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Extracting Landmarks from Videos](#extracting-landmarks-from-videos)
  - [Real-Time Indian Sign Language Recognition](#real-time-indian-sign-language-recognition)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The project consists of two main parts:
1. **Hand Landmark Extraction**: Extracts hand landmarks from videos and saves them as images and keypoint data files. This is useful for analyzing and training models for Indian Sign Language recognition.
2. **Real-Time Indian Sign Language Recognition**: Uses a pre-trained deep learning model to recognize Indian Sign Language in real-time from webcam input. The model predicts gestures based on extracted landmarks, utilizing functions from `testExtractLandmarksModel.py`.

## Features

- Extract and save hand landmarks from video files.
- Recognize Indian Sign Language in real-time using webcam input.
- Customizable settings for hand detection and recognition.
- Save extracted landmarks as images and `.npy` files for further analysis.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rf-uv-11/indian-sign-language-recognition.git
   cd indian-sign-language-recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have Python 3.6+ installed.

3. **Download or train your recognition model**:
   - Place your trained model (`indian_sign_language_classification_model.h5`) in the `model/` directory.

## Usage

### Extracting Landmarks from Videos

To extract hand landmarks from videos:

1. **Prepare the input videos**:
   - Place your videos in the `sign_language_videos` directory with subfolders for each letter (A-Z).

2. **Run the extraction script**:
   ```bash
   python extractImagesLandmarks.py
   ```
   This will process the videos, extract landmarks, and save them in `extracted_landmarks_images` and `extracted_landmarks_keypoints` directories.

### Real-Time Indian Sign Language Recognition

To run the real-time Indian Sign Language recognition:

1. **Ensure your model is in place**.
2. **Run the recognition script**:
   ```bash
   python recognize.py
   ```
   This script utilizes functions from `testExtractLandmarksModel.py` to perform landmark extraction and gesture recognition in real-time using a webcam feed. The recognized Indian Sign Language gestures will be displayed on the screen.

## Project Structure

```plaintext
.
├── extractImagesLandmarks.py        # Script for extracting landmarks from videos
├── recognize.py                     # Script for real-time Indian Sign Language recognition
├── testExtractLandmarksModel.py     # Script with functions for testing and extracting landmarks used by recognize.py
├── utils.py                         # Utility functions for the project
├── model/
│   └── indian_sign_language_classification_model.h5  # Pre-trained recognition model
├── sign_language_videos/            # Input videos directory
├── extracted_landmarks_images/      # Output directory for extracted landmark images
├── extracted_landmarks_keypoints/   # Output directory for extracted landmark keypoints
├── requirements.txt                 # List of dependencies
└── README.md                        # Project README file
```

## Dependencies

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- TensorFlow
- Additional dependencies listed in `requirements.txt`