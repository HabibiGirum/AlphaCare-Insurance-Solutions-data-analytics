import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

class StatisticalModeling:
    """
    Class for performing Statistical Modeling.

    Methods:
        __init__: Initializes the class with the dataset.
        prepare_data: Prepares the data by handling missing values, encoding, and splitting.
        build_models: Builds Linear Regression, Random Forest, and XGBoost models.
        evaluate_models: Evaluates the models using MSE and R-squared.
        interpret_model: Interprets the model using SHAP.
    """
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

   
        """
        Prepares the data for modeling by handling missing values, encoding, and splitting.
        """
        # Initial data check
        print("Initial data shape:", self.data.shape)
        print(self.data.isnull().sum())

        # Step 1: Impute missing values in 'TotalPremium' with the mean
        if 'TotalPremium' in self.data.columns:
            mean_value = self.data['TotalPremium'].mean()
            self.data['TotalPremium'].fillna(mean_value, inplace=True)
            print(f"Imputed missing 'TotalPremium' values with mean: {mean_value}")

        # Step 2: Impute missing values for numeric columns with their mean
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if self.data[col].isnull().any():
                mean_value = self.data[col].mean()
                self.data[col].fillna(mean_value, inplace=True)
                print(f"Imputed missing values in '{col}' with mean: {mean_value}")

        # Step 3: Impute missing values for categorical columns with the mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().any():
                mode_value = self.data[col].mode()[0] if not self.data[col].mode().empty else None
                if mode_value is not None:
                    self.data[col].fillna(mode_value, inplace=True)
                    print(f"Imputed missing values in '{col}' with mode: {mode_value}")

        # Check for remaining NaNs after imputation
        if self.data.isnull().sum().sum() > 0:
            print("Remaining NaN values in the dataset after imputation:")
            print(self.data.isnull().sum())
            raise ValueError("NaN values still exist in the dataset after preprocessing.")

        # Continue with feature engineering and data preparation
        # Step 4: Feature Engineering
        if 'RegistrationYear' in self.data.columns:
            self.data['AgeOfVehicle'] = 2024 - self.data['RegistrationYear']
            print("Added 'AgeOfVehicle' feature.")

        # Step 5: One-hot encoding for categorical data
        data_encoded = pd.get_dummies(self.data, columns=['IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 
                                                        'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Country', 
                                                        'Province', 'MainCrestaZone', 'SubCrestaZone', 'ItemType', 
                                                        'VehicleType', 'CoverCategory', 'CoverType', 'CoverGroup', 
                                                        'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType'], 
                                    drop_first=True)

        # Step 6: Define Features (X) and Target (y)
        if 'TotalPremium' not in data_encoded.columns:
            raise ValueError("Target variable 'TotalPremium' is missing after preprocessing.")

        X = data_encoded.drop(['TotalPremium', 'TotalClaims'], axis=1, errors='ignore')
        y = data_encoded['TotalPremium']

        if X.empty or y.empty:
            raise ValueError("No features or target left after preprocessing. Please check the data.")

        # Final sanity check
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            print("Remaining NaN values in features or target after preprocessing:")
            print(X.isnull().sum())
            print(y.isnull().sum())
            raise ValueError("NaN values still exist in the dataset after preprocessing.")

        print(f"Final dataset size: {X.shape[0]} rows, {X.shape[1]} columns")

        # Step 7: Train-test split
        if X.shape[0] < 2:
            raise ValueError("Not enough samples to split into train and test sets.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"Train set size: {self.X_train.shape[0]} rows, Test set size: {self.X_test.shape[0]} rows")
    def build_models(self):
        """
        Builds and fits Linear Regression, Random Forest, and XGBoost models.
        """
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr_model

        # Random Forest
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model

        # XGBoost
        xgb_model = XGBRegressor(random_state=42)
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model

    def evaluate_models(self):
        """
        Evaluates all models using Mean Squared Error and R-squared metrics.
        """
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"{name} - MSE: {mse:.2f}, R^2: {r2:.2f}")

    def interpret_model(self, model_name):
        """
        Interprets the given model using SHAP.
        Args:
            model_name (str): Name of the model to interpret.
        """
        model = self.models.get(model_name)
        if model is None:
            print(f"Model {model_name} not found.")
            return

        explainer = shap.Explainer(model, self.X_train)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test)

