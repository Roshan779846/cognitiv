# cognitive

Here's how you can navigate to the minor directory and run the EEGSDK.py file:

# EEG Data Processing and Machine Learning Model

This guide outlines the steps to process EEG data, extract features, train a machine learning model, and evaluate its performance. Additionally, it includes commands to navigate directories and run the EEG script.

## Prerequisites

- Python 3.x
- Required libraries: numpy, scipy, pandas, sklearn

## Directory Navigation and Running the EEG Script

To navigate to the `minor` directory and run the EEG script, use the following commands:


Change directory to "minor"

`cd minor`

Run the EEG file (assuming it is a Python script named eeg.py)

 `python eeg.py`


Future Improvements(Involves making ML model to improve accuracy)
For future improvements, consider including a machine learning model to train on this EEG data for accurate predictions. The steps would involve:

Data Preprocessing: Clean and preprocess the EEG data to make it suitable for machine learning.
Feature Extraction: Extract relevant features from the EEG signals.
Model Selection: Choose appropriate machine learning algorithms.
Training: Train the model using the extracted features and corresponding labels.
Evaluation: Evaluate the model's performance using relevant metrics.
Hyperparameter Tuning: Optimize the model's hyperparameters.
Deployment: Deploy the trained model for real--time predictions on new EEG data.

```bash


# EEG Data Acquisition and Emotion Detection

This project is an EEG data acquisition and emotion detection system using Python. The system connects to an EEG hardware device via a serial port, retrieves EEG data, and detects cognitive states and emotions using the data.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Functions](#functions)
- [EEG Data Processing and Machine Learning Model](#eeg-data-processing-and-machine-learning-model)
  - [1. Data Preprocessing and Feature Extraction](#1-data-preprocessing-and-feature-extraction)
  - [2. Model Training and Evaluation](#2-model-training-and-evaluation)
  - [3. Hyperparameter Tuning](#3-hyperparameter-tuning)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction

The EEGSDK is a Python-based SDK that connects to an EEG hardware device, retrieves EEG data in real-time, and processes the data to detect cognitive states and emotions. The data is also streamed using the Lab Streaming Layer (LSL) protocol for further analysis.

## Features
- Connect to EEG hardware via a serial port
- Start and stop data acquisition
- Retrieve and process EEG data in real-time
- Stream EEG data using LSL protocol
- Detect and plot cognitive states and emotions

## Requirements
- Python 3.6+
- numpy
- pylsl
- matplotlib
- pandas
- pyserial

## Setup

Clone the repository:

```sh
git clone https://github.com/your-username/eeg-emotion-detection.git
cd eeg-emotion-detection


INSTALL REQUIRED PACKAGES
pip install numpy pylsl matplotlib pandas pyserial



Usage
Update the serial_port parameter in the __main__ section of the script with the correct serial port your EEG hardware is connected to (e.g., COM7 for Windows or /dev/ttyUSB0 for Linux).

Run the script:

sh
Copy code
python eeg_sdk.py
The script will start data acquisition and plot the detected emotions in real-time. To stop the acquisition, press Ctrl+C.

Functions
__init__(self, serial_port, baud_rate=9600, lsl_stream_name='ORIC-EEG', json_file_path='eeg_data.json')
Initializes the EEGSDK object with the specified parameters.

connect(self)
Connects to the EEG hardware via the specified serial port.

disconnect(self)
Disconnects from the EEG hardware.

start_data_acquisition(self)
Starts data acquisition from the EEG hardware.

stop_data_acquisition(self)
Stops data acquisition from the EEG hardware.

retrieve_eeg_data(self)
Retrieves EEG data from the hardware, processes it to detect emotions, and streams the data using the LSL protocol.

plot_emotions(self)
Plots the detected emotions over time.

EEG Data Processing and Machine Learning Model
1. Data Preprocessing and Feature Extraction
python
Copy code
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to group columns by the first letter
def group_columns_by_first_letter(df):
    grouped_dict = {}
    for column in df.columns:
        key = column[0].lower()
        if key in grouped_dict:
            grouped_dict[key].append(df[column].tolist())
        else:
            grouped_dict[key] = [df[column].tolist()]
    return grouped_dict

# Functions for frequency filtering
def frequency_filter(data, frequency_band, fs):
    if frequency_band == "Alpha":
        return butter_bandpass_filter(data, 7.5, 12.5, fs)
    if frequency_band == "Beta":
        return butter_bandpass_filter(data, 11.5, 33.5, fs)
    if frequency_band == "Gamma":
        return butter_bandpass_filter(data, 32.5, 45.5, fs)
    if frequency_band == "Theta":
        return butter_bandpass_filter(data, 3.5, 8.5, fs)
    if frequency_band == "Delta":
        return butter_bandpass_filter(data, 0.1, 4, fs)
    else:
        return data

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a) * data[0]
    y, _ = lfilter(b, a, data, zi=zi)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to calculate power in frequency bands using Welch method
def calculate_power_bands(eeg_data, fs):
    frequency_bands = {
        'Delta': (0.1, 3),    # Delta band (0.5-4 Hz)
        'Theta': (4, 8),      # Theta band (4-8 Hz)
        'Alpha': (8, 12),     # Alpha band (8-12 Hz)
        'Beta': (12, 29),     # Beta band (12-33 Hz)
        'Gamma': (30, 45)     # Gamma band (33-63 Hz)
    }
    power_bands = {band: 0.0 for band in frequency_bands}
    for band_name, (low_freq, high_freq) in frequency_bands.items():
        f, Pxx = welch(eeg_data, fs=fs, nperseg=len(eeg_data))
        band_indices = np.where((f >= low_freq) & (f <= high_freq))
        band_power = np.trapz(Pxx[band_indices], f[band_indices])
        power_bands[band_name] = band_power
    return power_bands

# Function to preprocess data
def preprocess_data(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled

# Preprocess the data
X = preprocess_data(df)
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
2. Model Training and Evaluation
python
Copy code
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
3. Hyperparameter Tuning
python
Copy code
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)





