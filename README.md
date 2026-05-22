# ⚽ Football Goal-Prediction-Model

A Machine Learning project that predicts football player goals using player performance statistics and advanced regression modeling with **XGBoost Regressor**.

---

## 📌 Project Overview

This project uses football player statistics from the **2016 football season** to train a Machine Learning model capable of predicting player goals.

The dataset is split into:

- **80% Training Data**
- **20% Testing Data**

The final model is trained using:

- **XGBoost Regressor (`XGBRegressor`)**

to generate accurate football goal predictions.

---

## 🚀 Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## 📂 Dataset Features

The dataset contains football statistics such as:

| Feature | Description |
|---|---|
| Matches_Played | Number of matches played |
| Substitution | Substitution appearances |
| Mins | Minutes played |
| Goals | Actual goals scored |
| xG | Expected goals |
| Shots | Total shots |
| OnTarget | Shots on target |
| Shots Per Avg Match | Average shots per match |
| On Target Per Avg Match | Average on-target shots per match |

---

## 🧠 Machine Learning Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Train-Test Split (80/20)
       ↓
Model Training using XGBoost
       ↓
Model Evaluation
       ↓
Goal Prediction
       ↓
Visualization
```

---

## 📊 Data Visualization

Visualization was performed using:
- Matplotlib
- Seaborn

### Graphs Used
- Correlation Heatmap
- Goal Distribution Plot
- Actual vs Predicted Goals Plot
- Regression Plot

---

## 🤖 Model Used

### XGBoost Regressor

The project uses:

```python
XGBRegressor
```

because it performs exceptionally well on:
- Tabular datasets
- Nonlinear relationships
- Sports analytics prediction problems

---

## 📈 Model Evaluation Metrics

The model performance was evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 📉 Example Visualization

The model compares:
- Actual Goals
- Predicted Goals

using regression and line plots for performance analysis.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/goal-prediction-model.git
```

Move into the project directory:

```bash
cd goal-prediction-model
```

Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## ▶️ Run the Project

Run the Jupyter Notebook:

```bash
jupyter notebook
```

or run Python scripts:

```bash
python train.py
```

---

## 📁 Project Structure

```text
goal-prediction-model/
│
├── data/
│   └── Data.csv
│
├── notebooks/
│   └── model.ipynb
│
├── outputs/
│   └── predictions.csv
│
├── models/
│   └── xgboost_model.pkl
│
├── README.md
│
└── requirements.txt
```

---

## 🎯 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Engineering
- Deep Learning Models
- Flask/FastAPI Deployment
- Real-time Football Analytics Dashboard

---

## 📌 Key Learning Outcomes

This project helped in understanding:

- Regression Modeling
- Sports Analytics
- Data Visualization
- Model Evaluation
- Feature Engineering
- Machine Learning Workflow
- XGBoost Implementation

---

## 👨‍💻 Author

### Piyush Jadhav

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!