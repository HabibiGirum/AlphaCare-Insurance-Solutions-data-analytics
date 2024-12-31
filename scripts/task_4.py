# import pandas as pd
# import numpy as np
# from scipy.stats import ttest_ind, chi2_contingency
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import shap
# import matplotlib.pyplot as plt

# class StatisticalModeling:
#     def __init__(self, data):
#         self.data = data
#         self.target = "TotalPremium"
#         self.features = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None

#     def prepare_data(self):
#         # Handle missing values
#         self.data.fillna(self.data.median(), inplace=True)
        
#         # Feature Engineering
#         self.data["VehicleAge"] = 2024 - self.data["RegistrationYear"]
        
#         # Encoding Categorical Data
#         data_encoded = pd.get_dummies(self.data, drop_first=True)

#         # Train-Test Split
#         self.features = [col for col in data_encoded.columns if col != self.target]
#         X = data_encoded[self.features]
#         y = data_encoded[self.target]
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     def linear_regression(self):
#         lr = LinearRegression()
#         lr.fit(self.X_train, self.y_train)
#         y_pred = lr.predict(self.X_test)
#         print("Linear Regression MSE:", mean_squared_error(self.y_test, y_pred))
#         print("Linear Regression R2:", r2_score(self.y_test, y_pred))

#     def random_forest(self):
#         rf = RandomForestRegressor(random_state=42)
#         rf.fit(self.X_train, self.y_train)
#         y_pred = rf.predict(self.X_test)
#         print("Random Forest MSE:", mean_squared_error(self.y_test, y_pred))
#         print("Random Forest R2:", r2_score(self.y_test, y_pred))

#     def xgboost(self):
#         xgb = XGBRegressor(random_state=42)
#         xgb.fit(self.X_train, self.y_train)
#         y_pred = xgb.predict(self.X_test)
#         print("XGBoost MSE:", mean_squared_error(self.y_test, y_pred))
#         print("XGBoost R2:", r2_score(self.y_test, y_pred))

#     def feature_importance_analysis(self):
#         rf = RandomForestRegressor(random_state=42)
#         rf.fit(self.X_train, self.y_train)
#         explainer = shap.Explainer(rf, self.X_test)
#         shap_values = explainer(self.X_test)

#         shap.summary_plot(shap_values, self.X_test)

