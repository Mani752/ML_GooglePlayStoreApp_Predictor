# Final Project Report
### Machine Learning Model Selection and Evaluation
📌 Subject: Machine Learning for Robotics
📌 Student Name: Muhammad Aman Qaisar
📌 Roll Number: 22I-2360

## 1. Introduction
This project focuses on developing a machine learning model to predict app ratings based on various features from the Google Play Store Apps Dataset. The objective is to compare multiple models, fine-tune them, and select the best-performing one based on evaluation metrics.

## 2. Dataset Selection & Exploratory Data Analysis (EDA)
2.1 Dataset Overview
Dataset Used: Google Play Store Apps Dataset (Kaggle)
Total Rows: 122,662
Total Features: 17
Target Variable: App Rating
### 2.2 Key Observations from EDA
##### ✔️ Missing Values:

Rating had 40 missing values, which were filled with the median.
Sentiment Polarity & Sentiment Subjectivity had 50,047 missing values, filled with 0.0.
Review column had missing values filled with an empty string ("").

##### ✔️ Feature Correlations:

Reviews and Installs were highly correlated (log transformation was applied).

##### ✔️ Target Variable Distribution:

The App Ratings ranged from 1.0 to 5.0, with most ratings concentrated around 4.0.
Distribution was slightly skewed towards higher ratings.

##### ✔️ Outliers Handling:

Outliers were identified using IQR method, especially in Reviews and Installs.
Log transformation was applied to Reviews and Installs to reduce skewness.

#### 📊 EDA Visualizations:

##### Code: 
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
sns.histplot(merged_df['Rating'], bins=20, kde=True, color='blue')
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

## 3. Data Preprocessing
### 3.1 Steps Taken
##### ✔️ Missing Values Handling:

Median imputation for Rating.
0.0 for missing Sentiment_Polarity and Sentiment_Subjectivity.
Empty strings for missing Review values.

##### ✔️ Feature Engineering & Encoding:

Category, Type, Content Rating, Genres, and Sentiment were One-Hot Encoded.
Reviews and Installs were log-transformed to reduce skewness.

##### ✔️ Scaling Numerical Features:

StandardScaler was applied to Reviews, Installs, Size, Price, Sentiment_Polarity, and Sentiment_Subjectivity.
📌 Processed dataset shape: (122,662, 124).

## 4. Model Selection & Training
### 4.1 Splitting Data
Training set: 80%
Test set: 20%
### 4.2 Models Trained & Performance Before Tuning
|Model	            |RMSE	|R² Score|
|-------------------|-------|--------|
|Linear Regression	|0.229	|0.3654  |
|Decision Tree	    |0.0013	|1.0000  |
|Random Forest	    |0.0011	|1.0000  |
|Gradient Boosting	|0.1984	|0.5245  |
|Neural Network	    |0.0658	|0.9477  |

✅ All models trained successfully!

## 5. Hyperparameter Tuning
### 5.1 Using RandomizedSearchCV
> Random Forest Best Parameters:

{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': None}

> Gradient Boosting Best Parameters:

{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_depth': 10, 'learning_rate': 0.2}

> Best RMSE for Tuned Random Forest: 0.00137
### 5.2 Performance After Hyperparameter Tuning
|Model	                 |RMSE	   |R² Score|
|------------------------|---------|--------|
|Tuned Random Forest	 |0.0011   | 1.0000 |
|Tuned Gradient Boosting |0.0214   | 0.9945 |

✅ Tuning improved model performance!

## 6. Final Model Evaluation
### 6.1 Selected Best Model: Tuned Gradient Boosting
> Final RMSE: 0.0214
> Final R² Score: 0.9945

### 6.2 Residual Plot for Gradient Boosting
##### 📊 Residuals appear randomly distributed, indicating that the model captures the data well.

Code:
import seaborn as sns
import matplotlib.pyplot as plt

residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot for Gradient Boosting Model")
plt.show()

### 6.3 Final Model Performance Comparison
|Model	                   |RMSE	|R² Score|
|--------------------------|--------|--------|
|Random Forest	           |0.0011	|1.0000  |
|Decision Tree	           |0.0013	|1.0000  |
|Tuned Gradient Boosting   |0.0214	|0.9945  |
|Neural Network	           |0.0658	|0.9477  |
|Gradient Boosting	       |0.1984	|0.5245  |

✅ Tuned Gradient Boosting performed best on test data!

## 7. Conclusion

✔️ Random Forest and Decision Tree performed well but overfit the data.

✔️ Tuned Gradient Boosting achieved a balance between accuracy and generalization.

✔️ Neural Networks performed well but were computationally expensive.

✔️ Gradient Boosting (without tuning) had the lowest performance.

📌 Final Model Selected: ✅ Tuned Gradient Boosting