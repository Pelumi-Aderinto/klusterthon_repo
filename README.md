# Precision Farming Model Deployment Detailed Documentation
This project is aimed at building a solution that predicts the best time to harvest crops, taking into account local weather conditions, crop information, and soil quality. This solution will help farmers improve their yields and help them farm with surgical precision. This repo details the link to the Front End code, Back End code, link postman documentation and finally the details of how the model was built. Please check the later section of this readme file to see the link to the frontend and backend codes as well as the link to the postman documentation. 

# Table of Contents
1. Introduction
1. Project Overview
1. Data Preparation
   * Data Collection
   * Exploratory Data Analysis (EDA)
   * Data Preprocessing
1. Model Training
   * Model Selection
   * Model Training Process
   * Model Evaluation
1. Model Serialization
   * Saving the Trained Model
1. Flask Application
   * Setting up Flask
   * API Endpoint for Prediction
1. Deployment on PythonAnywhere
   * Uploading Files to PythonAnywhere
   * Running the Flask App

# INTRODUCTION
This document provides a step-by-step guide for deploying and hosting a crop harvest season prediction model. The entire process covers data preparation, model training, serialization, and deployment using Flask on the PythonAnywhere platform.

# PROJECT OVERVIEW
The project involves predicting the best season to harvest a specific crop based on environmental factors and geographical location. A machine learning model is trained for predicting the harvest season.

# DATA PREPARATION
## DATA COLLECTION
The dataset is loaded from a CSV file named "Crop_Data.csv," containing information about environmental factors, crop labels, and geographical details.
## EXPLORATORY DATA ANALYSIS (EDA)
Exploratory Data Analysis is performed to understand the dataset, including visualizations and insights gained from the data. This EDA is done in the later section of the `klusterthon3.ipynb` notebook.
## DATA PREPROCESSING
Categorical variables such as 'label,' 'Country,' and 'harvest_season' are encoded using LabelEncoders. Numerical features are standardized using a StandardScaler.

# MODEL TRAINING
## MODEL SELECTION
Support Vector Classifiers (SVC) is chosen as the machine learning model for crop harvest season prediction for the following reasons/justification;
* Versatility: SVMs can be adapted to different problem types, including classification and regression, through the use of different kernels (linear, polynomial, radial basis function, etc.)
* Robust to Overfitting: SVMs have a regularization parameter (C) that helps control the trade-off between a smooth decision boundary and classifying training points correctly. This makes SVMs less prone to overfitting.
* Effective in Non-Linear Spaces: SVMs can handle non-linear decision boundaries by using non-linear kernels, allowing them to capture complex relationships in the data.
* Works Well with Small and Medium-Sized Datasets: SVMs often perform well when the number of features is not extremely large and when the dataset is not excessively large.

## MODEL TRAINING PROCESS
The model is trained using the SupportVectorClassifier(SVC) algorithm with an rbf kernel and a regularization parameter C = 1 (These parameters were chosen after a grid search of the best parameters to use all of which processes are included in the notebook). The dataset is split into training and testing sets in the 80%-20% ratio.

## MODEL EVALUATION
Model accuracy is evaluated for the crop harvest season prediction using sklearn's accuracy_score, precision_score, recall_score, f1_score, and classification_report. The model gave an accuracy of 91.07%

# MODEL SERIALIZATION
## SAVING THE TRAINED MODEL
The trained crop harvest season prediction model, along with LabelEncoders and the StandardScaler, are saved using joblib. LabelEncoders for 'season,' 'label,' 'Country,' and 'harvest_season' are saved for consistency during predictions. The StandardScaler used for numerical feature scaling is also saved.

# FLASK APPLICATION
## Setting up Flask
Flask is used to create a web application. The necessary libraries are imported.

## API Endpoint for Prediction
An API endpoint is created to receive input data and return crop harvest season prediction results. The endpoint includes standardization of input data using the saved StandardScaler and prediction using the trained models and LabelEncoders.

# DEPLOYMENT ON PYTHONANYWHERE
## Uploading Files to PythonAnywhere
Files, including the Flask application, trained models, LabelEncoders, and the StandardScaler, were uploaded to PythonAnywhere.

## Running the Flask App
The Flask application is finally run and hosted on PythonAnywhere.
Here is a link to the [endpoint](https://pelvic23.pythonanywhere.com/predict?temperature=17&humidity=160&ph=7.5&water_availability=80&label=chickpea&country=Nigeria)

# [LINK TO BACKEND CODE](https://github.com/oresho/smartfarm)

# [LINK TO FRONTEND CODE](https://github.com/Abdulsalam24/smartfarm-group)

# [LINK TO POSTMAN](https://documenter.getpostman.com/view/28605577/2s9YeD9ZNQ)

