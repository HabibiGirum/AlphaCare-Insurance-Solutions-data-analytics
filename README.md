# week-3  Exploratory Data Analysis (EDA) for Car Insurance Claims Dataset

![Build Status](https://github.com/HabibiGirum/AlphaCare-Insurance-Solutions-data-analytics/actions/workflows/unittests.yml/badge.svg)

## Overview

This project focuses on performing Exploratory Data Analysis (EDA) on a historical insurance claims dataset for AlphaCare Insurance Solutions (ACIS). The aim of this analysis is to help optimize the marketing strategy and identify "low-risk" targets where premiums can be reduced to attract new clients.

## Dataset

The dataset used in this analysis contains historical insurance claim data for car insurance policies. Below is a summary of key columns in the dataset:

- **UnderwrittenCoverID**: Unique identifier for the insurance coverage.
- **PolicyID**: Unique identifier for the insurance policy.
- **TransactionMonth**: Date of the transaction (policy start date).
- **IsVATRegistered**: Whether the policyholder is VAT registered.
- **Citizenship**: Citizenship status of the policyholder.
- **LegalType**: The legal type of the policyholder.
- **Gender**: Gender of the policyholder.
- **Country**: Country of residence for the policyholder.
- **PostalCode**: Postal code (Zip code) for the policyholder's address.
- **TotalPremium**: The total premium paid by the policyholder.
- **TotalClaims**: The total claims made by the policyholder.

The dataset includes both numerical and categorical features, and the analysis includes methods to explore the relationships between variables and trends over time and geography.

## Installation

To run the code in this repository, follow these steps:

### Prerequisites

Make sure you have Python 3.11. You can check your Python version by running:

```bash
python --version
```

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/HabibiGirum/AlphaCare-Insurance-Solutions-data-analytics.git
cd AlphaCare-Insurance-Solutions-data-analytics
```

### Create a Virtual Environment (Optional but Recommended)

Create a virtual environment to isolate the project dependencies:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### Install Required Packages

Install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

### Running the Analysis

Once you have set up the environment and installed the dependencies, you can run the EDA scripts:

```notbooks/analysis.ipynb```

## About the Analysis

This repository includes the following steps as part of the EDA:

1. **Data Cleaning**: Handling missing values and converting data types.
2. **Univariate Analysis**: Visualizing distributions of numerical and categorical variables using histograms, bar charts, and box plots.
3. **Bivariate Analysis**: Exploring relationships between `TotalPremium` and `TotalClaims`, and other variables, using scatter plots and correlation matrices.
4. **Geographic Comparison**: Analyzing the distribution of premiums and claims across different geographical locations.
5. **Outlier Detection**: Identifying and handling outliers in the dataset.
6. **Trend Analysis**: Examining trends over time, such as premium changes and claim frequencies.

## Further documentation :
[click me](https://drive.google.com/file/d/15aGTZZdOCfE5vhIW5yV4cRS2wzHES72a/view?usp=sharing)


## Author  
GitHub: [HabibiGirum](https://github.com/HabibiGirum)

Email:  habtamugirum478@gmail.com