import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

class ABHypothesisTesting:
    def __init__(self, data):
        self.data = data

    def perform_t_test(self, group_a, group_b, metric):
        return ttest_ind(group_a[metric], group_b[metric], equal_var=False)

    def test_province_risk_difference(self):
        group_a_province = self.data[self.data["Province"] == "ProvinceA"]
        group_b_province = self.data[self.data["Province"] == "ProvinceB"]
        
        stat, p_value = self.perform_t_test(group_a_province, group_b_province, "TotalClaims")
        if p_value < 0.05:
            print("Reject Null Hypothesis: Significant risk differences across provinces.")
        else:
            print("Fail to Reject Null Hypothesis: No significant risk differences across provinces.")

