# 🧠 XAI-Dashboard: Mental Health Prediction using Explainable AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning web application that predicts mental health treatment needs with **Explainable AI** capabilities, making black-box model decisions transparent and interpretable.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Explainability](#-explainability)
- [Dataset](#-dataset)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

Mental health conditions are often underdiagnosed due to stigma and lack of awareness. This project aims to:
- Predict whether an individual is likely to seek mental health treatment
- Provide **interpretable explanations** for predictions using XAI techniques
- Offer both batch processing (training) and single prediction modes

## ✨ Features

- **Data Preprocessing**: Automated handling of missing values, encoding categorical variables, and feature scaling
- **Machine Learning Model**: Random Forest classifier with optimized hyperparameters
- **Explainability**: Model interpretability using feature importance analysis
- **Flexible Input**: Process CSV files (batch mode) or single data points (prediction mode)
- **Model Persistence**: Save and load trained models, encoders, and scalers

## 📁 Project Structure
   xai-dashboard/
├── main.py # Main application entry point
├── model.py # ML model training and prediction logic
├── preprocessing.py # Data preprocessing pipeline
├── requirements.txt # Project dependencies
├── .gitignore # Git ignore rules
├── survey.csv # Raw survey data (optional)
├── cleaned_survey.csv # Processed dataset
├── mental_health_model.pkl # Trained Random Forest model
├── rf_model.pkl # Alternative model version
├── encoders.joblib # Saved label encoders
├── scaler.joblib # Saved feature scaler
└── xai_dashboard.code-workspace # VSCode workspace settings

## 🚀 Installation

1. **Clone the repository**
git clone https://github.com/Rakshita-Gummat/xai-dashboard-.git
cd xai-dashboard-
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
