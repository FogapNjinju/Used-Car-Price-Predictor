# Used-Car-Price-Predictor
A machine learning project to predict used car prices using a dataset from Kaggle. The project employs Random Forest and Gradient Boosting models with preprocessing, feature engineering, and hyperparameter tuning to achieve accurate predictions.

# Project Overview

This repository contains a Jupyter Notebook (UsedCarPrediction.ipynb) that implements a predictive model for used car prices. The dataset is sourced from Kaggle (Used Car Price Prediction) and includes features such as car brand, model, year, mileage, fuel type, and more.

# Objectives

- Build and evaluate machine learning models to predict used car prices.
- Perform data preprocessing, including handling missing values and encoding categorical variables.
- Optimize model performance through feature selection and hyperparameter tuning.
- Save the best-performing model for future use.

# Dataset
The dataset (Used_Car_DataSet.csv) contains the following key features:

- Id: Unique car identifier.
- Year: Manufacturing year.
- Brand: Car brand.
- Full_model_name: Detailed model name.
- Model_name: Simplified model name.
- Price: Target variable (sale price).
- Distance_travelled(kms): Mileage.
- Fuel_type: Fuel type (e.g., Petrol, Diesel).
- City: Registration city.
- Brand_rank: Brand popularity rank.
- Car_age: Age of the car.

# Requirements

To run the notebook, install the required Python packages:
pip install pandas numpy scikit-learn seaborn matplotlib joblib
Ensure you have Python 3.12.4 or compatible versions of the libraries (e.g., scikit-learn 1.4.2).

# Usage
Clone the repository:
git clone https://github.com/your-username/UsedCarPricePredictor.git
Place the dataset (Used_Car_DataSet.csv) in the project directory.
Open and run the Jupyter Notebook:
jupyter notebook UsedCarPrediction.ipynb
The notebook will:
Load and preprocess the data.
Train Random Forest and Gradient Boosting models.
Evaluate model performance using RMSE and R² metrics.
Save the best Gradient Boosting model as best_model_gb.pkl.
Display feature importance for the best model.

# Results
The Gradient Boosting model typically outperforms Random Forest, with key features like full_model_name and brand_rank being the most influential.
Model performance metrics (RMSE, R²) and feature importance are printed in the notebook output.
Future Improvements
Incorporate additional features (e.g., car condition, maintenance history).
Experiment with other models like XGBoost or neural networks.
Deploy the model as a web application for real-time predictions.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments
Dataset provided by Kaggle.


Built with Python, scikit-learn, and Jupyter Notebook.
