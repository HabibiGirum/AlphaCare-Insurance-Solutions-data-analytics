import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

class EDA:
    def __init__(self, df):
        """
        Initialize the EDA class with the provided dataset (df).
        """
        self.df = df
        
    def data_summary(self):
        """
        Perform a summary of the dataset including descriptive statistics and data structure.
        """
        print("Data Structure (Data Types):")
        print(self.df.dtypes)
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        print(self.df.columns.tolist())
        
    def data_quality_assessment(self):
        """
        Check for missing values and any potential data quality issues.
        """
        missing_data = self.df.isnull().sum()
        print("\nMissing Data:")
        print(missing_data[missing_data > 0])
        
    def univariate_analysis(self):
        """
        Perform univariate analysis to visualize distributions of numerical and categorical columns.
        """
        numerical_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        # Plot histograms for numerical columns
        self.df[numerical_columns].hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
        plt.suptitle('Histograms of Numerical Columns')
        plt.show()
        
        # Plot bar charts for categorical columns
        for col in categorical_columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, data=self.df, palette="Set2")
            plt.title(f'Countplot of {col}')
            plt.xticks(rotation=45)
            plt.show()
        
    def bivariate_analysis(self):
        """
        Perform bivariate analysis and explore relationships between variables.
        """
        # Ensure datetime columns are converted to datetime type
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            except Exception:
                pass  # Skip columns that cannot be converted to datetime
        
        # Select only numerical columns for correlation
        numerical_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.df[numerical_columns].corr()

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()

        # Scatter plot between TotalPremium and TotalClaims, grouped by PostalCode
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='PostalCode', data=self.df, palette="viridis")
        plt.title('TotalPremium vs TotalClaims by ZipCode')
        plt.show()


    def data_comparison(self):
        """
        Compare data based on geography and other categories.
        """
        # Drop rows with missing 'make' or 'TotalPremium'
        self.df = self.df.dropna(subset=['make', 'TotalPremium'])

        # Check if there are any 'make' categories with no data
        make_counts = self.df['make'].value_counts()
        valid_makes = make_counts[make_counts > 0].index
        
        # Filter out invalid 'make' categories
        self.df = self.df[self.df['make'].isin(valid_makes)]

        # Verify that we have valid data for boxplot
        print(f"Valid makes: {valid_makes}")

        # Trend analysis: Compare TotalPremium by Vehicle Make (or any relevant category)
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='make', y='TotalPremium', data=self.df)
        plt.title('Distribution of TotalPremium by Vehicle Make')
        plt.xticks(rotation=45)
        plt.show()



    def outlier_detection(self):
        """
        Detect outliers using boxplots for numerical features.
        """
        numerical_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_columns:
            if col == 'TotalPremium' or col == 'TotalClaims':
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.df[col], color='lightblue')
                plt.title(f'Boxplot for {col}')
                plt.show()

    def creative_visualizations(self):
        """
        Produce 3 creative visualizations capturing key insights.
        """
        # Visualization 1: Premium distribution by cover type
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='CoverType', y='TotalPremium', data=self.df, inner="quart", palette="coolwarm")
        plt.title('Distribution of TotalPremium by CoverType')
        plt.show()

        # Visualization 2: Claims frequency by VehicleType
        plt.figure(figsize=(12, 8))
        sns.countplot(x='VehicleType', data=self.df, palette="muted")
        plt.title('Frequency of Claims by VehicleType')
        plt.xticks(rotation=45)
        plt.show()

        # Visualization 3: Trends over time (e.g., TotalClaims over TransactionMonth)
        plt.figure(figsize=(12, 8))
        monthly_claims = self.df.groupby('TransactionMonth')['TotalClaims'].sum().reset_index()
        sns.lineplot(data=monthly_claims, x='TransactionMonth', y='TotalClaims', marker='o')
        plt.title('TotalClaims Trends Over TransactionMonth')
        plt.xticks(rotation=45)
        plt.show()
