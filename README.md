# COPD Prediction using ML

This repository contains the Machine Learning core of the COPD Prediction system. My role focused on the implementation of hybrid Deep Learning architectures for respiratory disease classification.

# My Core Contributions (ML & Data Analysis)
* Deep Learning Architecture: Designed and trained a Hybrid CNN + LSTM model:
    * CNN Layer: For spatial feature extraction from spectrograms.
    * LSTM Layer: To capture temporal dependencies in breathing cycles.
* Feature Engineering: Performed Exploratory Data Analysis (EDA) to isolate specific frequency ranges that differentiate healthy breathing from COPD patterns.
* Model Optimization: Achieved high precision by tuning hyperparameters and addressing class imbalance in the respiratory dataset.

# Tech Stack (ML Specialist)
* Programming: Python
* Deep Learning Frameworks: TensorFlow / Keras
* Audio Analytics: Librosa
* Data Science Libraries: NumPy, Pandas, Matplotlib, Seaborn
* Environment: Jupyter Notebook / Google Colab

# The Pipeline Developed
1. Extraction: Loading raw '.wav' files.
2. Transformation: Applying Short-Time Fourier Transform (STFT) to create Mel-spectrograms.
3. Normalization: Using NumPy to scale data into tensors ready for the Neural Network.
4. Classification: Passing data through the CNN+LSTM layers to output a probability score.

# Evaluation Metrics
* Focused on Recall and ROC-AUC scores, as missing a positive COPD case (False Negative) is critical in medical diagnostics.

[Dashboard Screenshot] - https://github.com/JebaMisra33-hub/COPD-Detection-ML-Chatbot/blob/main/ouyput_1.jpeg?raw=true
