# Berkeley_Module11_Cars
### Cource : Module 11 Berkeley
Link to my notebook: https://github.com/Minasardari/Berkeley_Module11_Cars/blob/main/MinaSardari_Car_price.ipynb
### business goal : 
The business goal is to identify the key features that influence used car prices, enabling a dealership to make informed purchasing and pricing decisions.

Business Understanding → Data Science Perspective:
From a data science perspective, this is a supervised regression problem, where the target variable is the car price, and the predictor variables include attributes such as region, condition, model, mileage (odometer), year, fuel type, and transmission.

Using the CRISP-DM framework, this project will begin with exploratory data analysis (EDA) to uncover patterns, trends, and relationships within the dataset. This will be followed by data cleaning and feature engineering to enhance model performance. Ultimately, regression models will be developed and evaluated to quantify the impact of each feature on car price, providing actionable insights to guide the dealership’s pricing strategy.

### Project Roadmap (CRISP-DM Steps):

#### 1. Understanding the Dataset
Grasp the structure, variables, and potential quality issues within the dataset.

#### 2. Initial Data Cleaning
Address missing values, duplicates, and incorrect data types; prepare the data for analysis.

#### 3. EDA – Univariate Analysis
Explore the distribution of individual variables to identify trends, outliers, and data quality issues.

#### 4. EDA – Bivariate and Multivariate Analysis
Investigate relationships between predictors and the target variable, as well as between predictors.

#### 5. Feature Engineering Ideas
Create or transform variables to better capture patterns and improve model performance (e.g., age from year, log-transform of mileage).

#### 6. Behavioral Insights & Hypotheses
Formulate hypotheses based on data patterns — e.g., "Newer cars with automatic transmission and low mileage have higher resale value."

#### 7. Model Building
Train multiple regression models (e.g., linear regression, decision trees, random forest) to predict car prices.

#### 8. Model Evaluation
Assess model accuracy using appropriate metrics (e.g., RMSE, MAE, R²) and validate generalization with train-test splits or cross-validation.

#### 9. Data Visualization for Storytelling
Develop intuitive charts and dashboards to communicate findings and support data-driven decision-making for the dealership.


### 1. Understanding the Dataset

Step 1: Load and Inspect the Data
df.head() – View the first few rows to understand structure.
df.info() – Check column names, data types, and non-null counts.
df.describe() – Review statistical summaries of numeric columns.
df.columns – Look for naming inconsistencies

Step 2: Identify Missing Values
# Calculate missing values and their percentage
![image](https://github.com/user-attachments/assets/a965a1b0-0857-43bc-93bb-eba7a1fa4738)

Step 3: explore data uniqe values for categorial features 
  Calculate proportions
  Replace NaN index labels with a string for display ''unknown
  Plot to see value counts

  
  ![image](https://github.com/user-attachments/assets/36e1c33b-7ca5-441e-a9ca-af477d44a249)
  ![image](https://github.com/user-attachments/assets/a56b8aaf-f237-433c-b34d-185a0eca3cc9)
  ![image](https://github.com/user-attachments/assets/38a2abdf-3d73-4d1e-ba1e-85bead642cbe)
  ![image](https://github.com/user-attachments/assets/7a416e65-8318-4993-8c08-59904367e888)
  ![image](https://github.com/user-attachments/assets/3f61787b-0c97-4b9b-b697-ab81ad91f436)
  ![image](https://github.com/user-attachments/assets/cce11a8e-2ef9-46f7-b446-072bb6a58f8f)

#### 2. Initial Data Cleaning
step 1: Drop Duplicate
Step 2: Drop Columns 
    VIN  not analytically useful for modeling.
    Drop Size too much missing data about 70% and we can use Type 
step 3: Categorical Features Imputation:
   Use new category "unknown" for columns [condition, drive, type, and paint_color] to preserve data volume.
   Use mode imputation for fuel, title_status, transmission.
   Define the critical columns critical_cols = ['model', 'manufacturer', 'odometer', 'year'] and drop rows where ALL critical columns are missing
   Fill Model with Mode based on Manufacturer and Cylinder if not just Manufacturer
step 4: Detect Data Quality Issues   
Invalid or inconsistent values for column Model
   df.loc[
    df['model'] == "$362.47, $1000 down, oac, 2.9%apr $362.47,luxury low miles $1000 down, only 40k miles",
    'model'
] = np.nan

Step 5: Numerical Features Imputation:
![image](https://github.com/user-attachments/assets/f766b66b-8d16-45d5-a8b9-284baa90552d)
Use median for odometer.
drop year if manufacturer and model missing
Find outlers and wrong data for Price and Year and Odometer and keep data as below:
``````python
price_cap_max = df['price'].quantile(0.99)
price_cap_min = df['price'].quantile(0.10)

vehicle_age_cap_max = df['year'].quantile(0.99)
vehicle_age_cap_min = df['year'].quantile(0.05)

odometer_cap_max = df['odometer'].quantile(0.95)
odometer_cap_min = df['odometer'].quantile(0.05)
``````
