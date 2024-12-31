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

    def prepare_data(self):
        """
        Prepares the data for modeling by handling missing values, encoding, and splitting.
        """
        # Step 1: Drop rows with NaN in the target column (TotalPremium)
        if 'TotalPremium' in self.data.columns:
            initial_rows = self.data.shape[0]
            self.data = self.data.dropna(subset=['TotalPremium'])
            print(f"Dropped {initial_rows - self.data.shape[0]} rows with missing 'TotalPremium'")
        
        # Step 2: Handle datetime columns and extract features
        for column in self.data.select_dtypes(include=['datetime', 'object']):
            try:
                self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
                if pd.api.types.is_datetime64_any_dtype(self.data[column]):
                    self.data[f'{column}_year'] = self.data[column].dt.year
                    self.data[f'{column}_month'] = self.data[column].dt.month
                    self.data[f'{column}_day'] = self.data[column].dt.day
                    self.data.drop(column, axis=1, inplace=True)
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                self.data.drop(column, axis=1, inplace=True)
        
        # Step 3: Handle excessive missing values
        initial_rows = self.data.shape[0]
        self.data = self.data.dropna(axis=0, thresh=int(0.7 * self.data.shape[1]))  # Keep rows with at least 70% non-NA
        print(f"Dropped {initial_rows - self.data.shape[0]} rows with excessive missing values")

        self.data.dropna(inplace=True)  # Drop remaining NaNs
        
        if self.data.empty:
            raise ValueError("No data left after preprocessing. Please check your dataset.")
        
        # Step 4: Feature Engineering
        if 'RegistrationYear' in self.data.columns:
            self.data['AgeOfVehicle'] = 2024 - self.data['RegistrationYear']
        
        # Step 5: One-hot encoding for categorical data
        data_encoded = pd.get_dummies(self.data, drop_first=True)
        
        # Step 6: Drop non-numeric columns
        non_numeric_columns = data_encoded.select_dtypes(include=['object']).columns
        if len(non_numeric_columns) > 0:
            print(f"Warning: Dropping non-numeric columns: {non_numeric_columns}")
            data_encoded.drop(non_numeric_columns, axis=1, inplace=True)
        
        # Step 7: Define Features (X) and Target (y)
        if 'TotalPremium' not in data_encoded.columns:
            raise ValueError("Target variable 'TotalPremium' is missing after preprocessing.")
        
        X = data_encoded.drop(['TotalPremium', 'TotalClaims'], axis=1, errors='ignore')
        y = data_encoded['TotalPremium']
        
        if X.empty or y.empty:
            raise ValueError("No features or target left after preprocessing. Please check the data.")
        
        # Final sanity check
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            raise ValueError("NaN values still exist in the dataset after preprocessing.")
        
        print(f"Final dataset size: {X.shape[0]} rows, {X.shape[1]} columns")
        
        # Step 8: Train-test split (only if enough samples exist)
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

        explainer = shap.Explainer(model, self.X_test)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test)



