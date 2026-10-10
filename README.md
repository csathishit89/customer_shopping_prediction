# Customer Shopping Revenue & Sales Prediction 📈🛒

A Machine Learning pipeline and predictive modeling project designed to forecast customer shopping sales and revenue using regression algorithms.

---

## 📁 Repository Structure

```text
customer_shopping_prediction/
│
├── Customer_Shopping_Prediction.ipynb    # Main Jupyter Notebook for EDA, training, and evaluation
├── sales_prediction_dt_model.pkl         # Trained Decision Tree regression model
├── sales_prediction_knn_model.pkl        # Trained K-Nearest Neighbors regression model
├── sales_prediction_rf_model.pkl         # Trained Random Forest regression model
└── README.md                             # Project documentation
```

---

## 📌 Project Overview

This repository focuses on building, tuning, and deploying machine learning models to predict future sales and revenue based on customer shopping behavior and historical transaction patterns.

### Key Objectives:
* **Exploratory Data Analysis (EDA)**: Uncovering patterns, feature correlations, and distributions affecting customer spending.
* **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numerical variables.
* **Model Training & Evaluation**: Training multiple regression algorithms and evaluating performance using metrics such as RMSE, MAE, and $R^2$ Score.
* **Model Persistence**: Serializing trained models using `pickle` for inference and deployment.

---

## 🤖 Trained Models Included

* **Decision Tree Regressor** (`sales_prediction_dt_model.pkl`): Captures non-linear relationships through recursive data partitioning.
* **K-Nearest Neighbors Regressor** (`sales_prediction_knn_model.pkl`): Predicts sales based on feature similarity among neighboring data points.
* **Random Forest Regressor** (`sales_prediction_rf_model.pkl`): An ensemble method that improves generalization and minimizes overfitting by averaging multiple decision trees.

---

## 🛠️ Requirements & Tech Stack

Ensure you have the following libraries installed to run the notebook and load the models:

* **Language**: Python 3.8+
* **Data Manipulation**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn`
* **Model Persistence**: `joblib` / `pickle`
* **Visualization**: `matplotlib`, `seaborn`

To install the required dependencies, run:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/csathishit89/customer_shopping_prediction.git
   cd customer_shopping_prediction
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Customer_Shopping_Prediction.ipynb
   ```

3. **Load and Test Saved Models**:
   You can load any of the pre-trained `.pkl` models in Python for inference:
   ```python
   import pickle

   # Load the Random Forest model
   with open('sales_prediction_rf_model.pkl', 'rb') as file:
       model = pickle.load(file)

   # Example prediction
   # predictions = model.predict(X_new)
   ```

---

## 👤 Author

* **Sathish Chandrasekar**
  * GitHub: [@csathishit89](https://github.com/csathishit89)