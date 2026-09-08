# Predicting House Prices

## 📌 Project Overview

This project focuses on developing a machine learning model to predict residential house prices based on a 
range of property characteristics.  
The project is based on a 
[Kaggle house-price prediction competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview),
where the objective is to build a model capable of accurately estimating the sale price of a house from historical property data.

The project follows an end-to-end machine learning workflow, including data exploration, data cleaning, feature engineering, 
handling missing values, categorical feature encoding, feature transformation, model development, hyperparameter tuning, and
model evaluation. Different machine learning algorithms are trained and compared to identify the approach that provides the 
best predictive performance.

## 🎯 Problem Statement

The objective of this project is to develop a machine learning model that accurately predicts the sales price of residential properties 
based on their available characteristics such as:  
- LotFrontage: Linear feet of street connected to property
- LandContour: Flatness of the property
- OverallQual: Overall material and finish quality
- YearBuilt: Original construction date
- Foundation: Type of foundation
- BsmtFinType1: Quality of basement finished area

The model must predict the corresponding value of the SalePrice variable, with the goal of producing accurate predictions that perform 
well according to the Kaggle competition's evaluation metric.

## 📊 Dataset

Kaggle provided the data that is used to build the model.  

Access the dataset from [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

## 📁 File Descriptions

| File/Folder | Description |
| --- | --- |
| models/ | Contains the notebooks for modelling based on different algorithms. |
| processed_data/ | Training and test data, csv files after processed in p_house_prices.ipynb |
| submissions/ | Predictions on the test set for submission to Kaggle. |
| data_description.txt | Detailed explanation on the columns in the dataset. |
| p_house_prices.ipynb | Perform EDA and Process the data. |
| sample_submission.csv | Sample submission from Kaggle. | 
| test.csv | Testing data |
| train.csv | Training data |

## 🔎 Exploratory Data Analysis

Findings: 
- Training data has 1460 rows and 81 columns
- Test data has 1459 rows and 80 columns
- Dataset contains houses sold between 2006 and 2010
- SalePrice is right-skewed, as it increases the amount of houses sold decreases.
- GrLivArea has two outliers
- Only 7 house have a pool
- Houses with 1 kitchen above grade have the highest prices
- Some columns are integers but are really categorical columns.

## 🛠️ Data Preprocessing

In ```p_house_prices.ipynb```:
- Separating categorical and numerical columns:  
  Analyzed the columns in the dataset to determine if the data they contain is categorical or numerical.
- Handle null values:  
For categorical features, NA values that mean the feature is not in the house were filled with the string 'FNA' 
while where the value was not recorded it was filled with the most frequent value.   
For numerical features, filled null values with average.
- Scale numeric features using the StandardScaler
- Encode categorical columns with OneHotEncoder

In ```processed_data\xgboosted_mod.ipynb``` which produced the best model:
- Handle Missing values:  
For categorical features where NA means the feature is not available in the house, filled with 'None'  
Filled features with 0, median or mode depending on what it represents.
- Encode categorical features with LabelEncoder and one hot encoding(pandas get_dummies())
- Carry out box-cox transformation on numerical features to reduce skewness
- Log transform SalePrice to have a normal distribution

## ⚙️ Feature Engineering

In ```processed_data\xgboosted_mod.ipynb```, ```TotalSF``` was created to calculate the total square feet of the house. 

## 🤖 Models

The following are the scores obtained from Kaggle for the different algorithms used to predict the SalePrice.
Root Mean Squared Logarithmic Error(RMSLE) was used to evaluate the model performance.

| ML Algorithm | Kaggle Score |
| --- | --- |
| Linear Regression | 0.18386 |
| Ridge | 0.15399 |
| Lasso | 0.15403 |
| Elastic Net | 0.15245 |
| Decision Tree Regressor | 0.19354 |
| Random Forest Regressor | 0.15328 |
| XGB Regressor | 0.12992 |

The xgboost model attained the highest score. 

## 🚀 How to Run the Project

1. Clone the repository
   ```
   https://github.com/gitcodmax/Predicting-House-Prices.git
   cd "project_folder"
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   ```
   Activate it on Windows:
    ```
   venv\Scripts\activate
    ```

3. Install dependencies
   ```
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

4. Obtain the dataset
   
    Download the dataset from the Kaggle competition:  
    [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

5. Run the notebooks  
Open the notebooks using Jupyter Notebook or VS Code and execute them.

## 📚 What I Learned

Linear models tend to perform better when the numerical features and target variable are 
approximately normally distributed. During this project, transforming skewed features helped
make their distributions more symmetric, allowing linear models to better capture the underlying 
relationships between the predictors and house prices. This highlighted the importance of checking 
feature distributions and applying appropriate transformations before training linear regression-based models.

```                                                MMAX CODES                                              ```
