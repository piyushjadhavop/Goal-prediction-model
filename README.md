# Football Goal Prediction Using Machine Learning

## Project Overview

The Football Goal Prediction project is a Machine Learning-based application designed to predict the number of goals a football player may score based on different performance statistics. The project uses historical football season data and applies regression techniques to analyze player performance and generate goal predictions.

The main objective of this project was to understand the complete Machine Learning workflow, including data preprocessing, feature engineering, model training, evaluation, visualization, and frontend deployment using Streamlit.

---

# Objectives

* Predict football player goals using statistical data.
* Learn practical Machine Learning workflow.
* Build an interactive frontend for user input and predictions.
* Visualize player statistics and prediction outputs.
* Deploy a user-friendly prediction system using Streamlit.

---

# Technologies Used

## Programming Language

* Python

## Libraries & Frameworks

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Streamlit
* Plotly
* Joblib
* UV (Python Package Manager)
* UV Virtual Environment (uv venv)

---

# Dataset Description

The dataset contains football player performance statistics from the 2016 football season.


Data was collected from Kaggle and used for analysis in this project.

### Features Used

* Matches Played
* Minutes Played
* xG (Expected Goals)
* xG Per Average Match
* Shots
* Shots On Target
* Shots Per Average Match
* On Target Per Average Match

### Target Variable

* Goals

---

# Project Workflow

## 1. Data Collection

Football player statistical data was collected and loaded into a Pandas DataFrame for analysis and preprocessing.

---

## 2. Data Preprocessing

Several preprocessing steps were performed:

* Handling missing values
* Cleaning column names
* Feature selection
* Data formatting
* Train-test splitting

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

---

## 3. Exploratory Data Analysis (EDA)

EDA was performed using:

* Matplotlib
* Seaborn

Visualizations included:

* Correlation heatmaps
* Distribution plots
* Feature relationship analysis
* Actual vs Predicted goal comparison

---

## 4. Model Training

The model was trained using:

### XGBoost Regressor (`XGBRegressor`)

XGBoost was selected because:

* It performs well on tabular datasets.
* It handles nonlinear relationships effectively.
* It provides high prediction accuracy.
* It reduces overfitting compared to many traditional models.

---

## 5. Model Evaluation

The model was evaluated using regression metrics such as:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score (Accuracy)

These metrics helped measure prediction accuracy and model performance.

---

# Visualization

Graphs and visualizations were created to compare:

* Actual goals
* Predicted goals
* Player performance statistics

Libraries used:

* Seaborn
* Matplotlib
* Plotly

---

# Frontend Development

An interactive frontend was developed using Streamlit.

### Features of the Frontend

* User-friendly input fields
* Real-time goal prediction
* Interactive data visualization
* Dynamic bar charts
* Prediction display system

Users can enter player statistics and instantly receive predicted goal outputs.

---

# Challenges Faced

During the project, several issues were encountered and solved, including:

* Feature selection errors
* Shape mismatch issues
* Data scaling confusion
* Visualization debugging
* Streamlit integration problems
* Prediction formatting issues

These challenges improved debugging and problem-solving skills.

---

# Key Learnings

This project helped in understanding:

* End-to-end Machine Learning workflow
* Regression modeling
* XGBoost implementation
* Data preprocessing
* Feature engineering
* Model evaluation
* Frontend integration using Streamlit
* Data visualization techniques
* Git and GitHub project management

---

# Future Improvements

Possible future enhancements include:

* Adding multiple football seasons
* Hyperparameter tuning
* Cross-validation
* Deep learning implementation
* Real-time football analytics dashboard
* Cloud deployment
* Player comparison system
* Team performance prediction

---

# Conclusion

The Football Goal Prediction project successfully demonstrates how Machine Learning can be applied in sports analytics to predict player performance using statistical data.

The project combines:

* Data Science
* Machine Learning
* Visualization
* Frontend Development

into one complete end-to-end application.

This project also strengthened practical skills in Python, model building, debugging, data analysis, and deployment workflows.

---

# Author

Piyush Rajesh Jadhav
