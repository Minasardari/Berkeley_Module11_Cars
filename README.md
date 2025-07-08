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
#### 3. EDA – Univariate Analysis

![image](https://github.com/user-attachments/assets/d41e9840-8908-4910-941a-4cfd5d502588)
![image](https://github.com/user-attachments/assets/7af03991-7ba6-4172-aac8-3936513417df)
![image](https://github.com/user-attachments/assets/7b5b25b1-56fe-4815-9eb1-069872dd1f62)
![image](https://github.com/user-attachments/assets/eeb6e333-c1b9-4ddb-8cd8-24e4720e5749)
![image](https://github.com/user-attachments/assets/03d9b89f-2707-42de-a956-8dd1056ac73f)
Finding for Numinal feature 
Price and Odometer are right skewed
year is left skewed
Odometer Box plot shows almost normal distrbution
Price medium is toward left and it makes sense

For Categoral Feature rerun the plots to see new data distrubution after imputing 
Finding: 
 We have over 500 Manufacturare/model but the Top 10 or Top 20 has most quantity
 Condition: The Good is most or mode for this feature means people usually have good are selling
 Silverado is most car for sale and Ford and Chevrolet are top cars in manufacturer
Condition has some unknown that based on plot we can replace with mode which is good condtion
Cylinders the 6 cylenders seems top count and some unknown probably can find from type but I decided to ignore
As was seen gas is most popular and Sedan type is popular , automatic trasmittion also top, drive 4wd is most popular and we can impute the unknown with 4wd
Paint color White and Black are top 

#### 4. EDA - Bivariate and Multivariate Analysis
Scatter plot with regression line  for Price and odometer and Price and Year
![image](https://github.com/user-attachments/assets/14a22ade-19d5-4af7-8190-e9b4f2446194)
![image](https://github.com/user-attachments/assets/3d192014-46d0-4a4f-8a5c-15f1cacdcec7)

As I was expecting the Price and Odometer has clearly slopes downward, confirming the inverse relationship
and the Price and year has clearly slopes upward or age downward, confirming the inverse relationship with age of car

Boxplot of price across 'condition', 'type','cylinders','fuel','title_status','drive','paint_color','transmission'
I was able to get some hints about noise from boxplot and outliers for price for each category

![image](https://github.com/user-attachments/assets/91ad2394-c837-4b2b-a7a8-72717fcfa691)

Good" condition has higher median than "excellent"
"Unknown" condition has a wide range and high median
![image](https://github.com/user-attachments/assets/10e18562-8cb2-4712-9344-76d716e43bc1)
![image](https://github.com/user-attachments/assets/53001c5c-2587-4990-ad05-c3c458b57b3a)
![image](https://github.com/user-attachments/assets/63ad95f6-8079-4882-8ed4-b6c3b95dda15)
![image](https://github.com/user-attachments/assets/52ee4359-a4a8-4722-a87b-8816f5ff019c)
 ![image](https://github.com/user-attachments/assets/d1a26db2-b648-4188-a5c0-eaeaa6879efc)
![image](https://github.com/user-attachments/assets/95a58837-cc7d-4c7c-baf5-edc472a92d00)
![image](https://github.com/user-attachments/assets/73ecba1e-0777-478d-a5bf-1b18a8390ffa)
![image](https://github.com/user-attachments/assets/bec0e23e-20be-43de-8fff-85cb030ae272)


we can eliminate outlers for some like other is transmission
``````python
df['transmission'].value_counts()
# Use IQR method or z-score to identify extreme low outliers
Q1 = df[df['transmission'] == 'other']['price'].quantile(0.25)
Q3 = df[df['transmission'] == 'other']['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
# we have some outlier in othere and less than q1 we can remove them
df = df[~((df['transmission'] == 'other') &
          ((df['price'] < lower_bound)))]

``````
Multivariate Analysis
#Add Vehicle Age and remove year and build correlation matrix an paire plot


![image](https://github.com/user-attachments/assets/93026b53-8464-48ce-a374-27fc514baf51)
![image](https://github.com/user-attachments/assets/ece92fc4-d48b-4e84-a7e4-efef78302adf)

### Feature Engineering Ideas
Step 1 : to remove the skewed from price and odometer we get log of them 
Step 2: Remove outlier per group for categorial features
Step 3: PCA and Scree Plot
![image](https://github.com/user-attachments/assets/aad14cc5-2c29-427c-a6b6-5def23e54ab1)
based on the scree plot decided to include both columns (year age and odometer)and not dropping any

 ### Behavioral Insights & Hypotheses
 Understanding the behavior for data can help better choise
 The box plot for price and condition is intresting as the Good seems higher mediem than Excellent and New and Like New maybe inverntory data is based has less top brands

 Box plot for Type and Price shows high varience for Sedan and Pickuip and Truck are top median
 Price by fuel has more median in disel and it is expected
 Price and Manufacturer for top 10 shows amost same price and this is intresting


 ### Model Building
  Manufacturer an model can be complex I choose top 10 manufacturer and decide use type not model
  Define Model based on State California, we can define same model on another states
  Target is Price and the feature are divided to 3 groups
 ``````
 numeric_features = ['vehicle_age', 'odometer']
 categorical_features = [
    'manufacturer', 'cylinders', 'fuel',
     'transmission', 'drive', 'type'
  ]

  ordinal_features = ['condition','title_status']
  target = 'price'
  ``````

I have decided to use two advanced forms of linear regression:

- Ridge Regression (prevents overfitting by penalizing large coefficients)
- Lasso Regression (also prevents overfitting but can eliminate irrelevant features)

- I used pipelines to:
- Apply feature transformations (like encoding text data and scaling numbers)
- Fit the Ridge or Lasso model in one consistent, repeatable process
- using Hot encode and Ordinal Encode and standardscaler in column transformer
- applying GridSearchCV for Ridge to fine the best alpha

### Model Evaluation

📊 Model Evaluation Summary
🔧 Ridge Regression with Polynomial Features
GridSearchCV was used to tune:

degree of polynomial features

alpha (regularization strength)

✅ Best Hyperparameters:

Degree: 2

Alpha: 1

🧮 Performance:

Train MSE: 0.1215

Test MSE: 0.1225

Test R² Score: -0.1225 → ❌ Worse than baseline

📌 Interpretation:
Model overfits mildly (train ≈ test error, but R² < 0).

Adding polynomial features may have introduced non-informative complexity.

The feature signal might be weak or too noisy.

🔧 Lasso Regression
Used with default or manually set alpha (assumed 0.1 or similar).

❌ No GridSearchCV tuning

🧮 Performance:

Train MSE: 0.2710

Test MSE: 0.2724


📌 Interpretation:
Lasso aggressively shrinks coefficients, possibly dropping too many predictors.
Without tuning alpha, it may be underfitting.


Model	Observations	Suggestions
Ridge	Slight overfitting, R² < 0	we can try simpler features (drop polynomial), or include more informative variables
Lasso	Higher error, likely underfit	Use GridSearchCV to tune alpha — Lasso is sensitive to this

### Data Visualization for Storytelling
##### 🚗 Average Price by Car Condition Clearly communicates the negative relationship between mileage and price.

![image](https://github.com/user-attachments/assets/dd6793e7-370f-4b76-8ee5-d58ffb16fc2c)

##### 📉 Odometer vs. Price with Trendline

![image](https://github.com/user-attachments/assets/0084747e-5a7d-451e-bb82-3b73ebe3225d)
![image](https://github.com/user-attachments/assets/81e5898e-4bfd-489d-9bf8-d7a36d946fe5)

##### 🏷️Top 20 Manufacturers by Listing Frequency,Helps dealership focus on the most common and popular brands.

![image](https://github.com/user-attachments/assets/87f05574-59e3-43d7-ab30-ec7f30a9a76b)


##### 🧠Feature Importance

![image](https://github.com/user-attachments/assets/8e02c772-8ef4-4919-8d03-a258bf651f38)


##### 📈 Predicted vs. Actual Prices
![image](https://github.com/user-attachments/assets/253eaeda-c3c7-4ce5-9ead-d2e8c224576f)


Next Steps & Strategic Recommendations

1)Integrate Insights into Inventory Management
Action: Prioritize acquiring vehicles that:

- Are in “like new” or “excellent” condition
- Have low mileage
- Belong to top-performing manufacturers e.g Ram

2) Refine Data Collection & Standardization
 Clean and consolidate high-cardinality fields like manufacturer and model during data entry or ingestion
 Less group per categoral feature e.g instead of new and like new and excellent just one group excellent
3) Implement model for each state and or each manufactuer and build a dashboard

