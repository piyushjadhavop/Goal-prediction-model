# ⚽ Football Goal Prediction System

## About the Project

Football clubs and analysts often use player statistics to evaluate performance and identify potential goal scorers. However, manually analyzing large amounts of player data can be time-consuming and may not always reveal hidden patterns.

This project uses Machine Learning to predict the number of goals a football player is likely to score based on key performance metrics such as minutes played, expected goals (xG), shots, and shots on target.

The model was trained using historical football player statistics and deployed through an interactive Streamlit application where users can enter player statistics and instantly receive goal predictions.

---

## Problem Statement

Predicting player goal output is an important challenge in football analytics. While metrics such as shots, xG, and minutes played provide valuable insights, understanding how these factors collectively influence goal scoring requires a data-driven approach.

The goal of this project is to build a Machine Learning model capable of learning patterns from historical player performance data and accurately predicting future goal outcomes.

---

## Features

* Predict football player goals using Machine Learning
* Interactive Streamlit web application
* Real-time goal predictions
* User-friendly input interface
* Visual representation of player statistics
* Automatic storage of user prediction data
* Model trained using XGBoost Regression

---

## Dataset

The project uses football player performance statistics including:

* Matches Played
* Minutes Played
* xG Per Average Match
* Shots
* Shots On Target
* Shots Per Average Match
* On Target Per Average Match

### Target Variable

* Goals

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Streamlit
* Plotly
* Joblib

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Cleaned dataset
* Handled missing values
* Selected important features
* Prepared training and testing datasets

### 2. Train-Test Split

The dataset was split into:

* 80% Training Data
* 20% Testing Data

### 3. Model Training

The model was trained using:

* XGBoost Regressor (XGBRegressor)

XGBoost was selected because it provides excellent performance for tabular datasets and regression tasks.

### 4. Model Evaluation

The model was evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## Streamlit Application

The application allows users to:

1. Enter football player statistics.
2. Generate goal predictions instantly.
3. View prediction results.
4. Visualize player performance data.
5. Store new prediction records for future analysis.

---

## Project Structure

```text
Football-Goal-Prediction/
│
├── app.py
├── model.pkl
├── data.csv
├── requirements.txt
├── uv.lock
├── pyproject.toml
└── README.md
```

---

## How to Run the Project

### Clone the Repository

```bash
git clone <repository-url>
```

### Navigate to the Project Folder

```bash
cd Football-Goal-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

---

## Future Improvements

Some possible enhancements include:

* Adding multiple football seasons
* Hyperparameter tuning
* Cross-validation
* Team-level predictions
* Player comparison dashboard
* Cloud deployment
* Real-time football analytics integration

---

## What I Learned

Through this project, I gained hands-on experience with:

* Data preprocessing
* Feature selection
* Regression modeling
* XGBoost implementation
* Data visualization
* Streamlit application development
* Model deployment
* Git and GitHub workflows

---

## Author

**Piyush Jadhav**

Machine Learning Enthusiast | Data Analytics Learner | Python Developer
