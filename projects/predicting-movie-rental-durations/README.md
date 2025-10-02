# 📀 Predicting DVD Rental Duration

## Project Overview
Regression modeling project to predict the number of days customers will rent DVDs, enabling a rental company to optimize inventory planning and resource allocation.

## 📊 Business Objective
A DVD rental company needs to predict rental duration based on customer and movie features to improve inventory management efficiency. The goal is to build a regression model achieving a Mean Squared Error (MSE) of 3 or less on the test set.

## 🎯 Project Requirements

### Data Preprocessing
1. **Feature Engineering**: Create `rental_length_days` from date columns
2. **Dummy Variables**: Extract features from `special_features` column
   - `deleted_scenes`: Binary indicator for "Deleted Scenes"
   - `behind_the_scenes`: Binary indicator for "Behind the Scenes"
3. **Feature Selection**: Identify predictive features while avoiding data leakage
4. **Train-Test Split**: 80-20 split with `random_state=9`

### Model Requirements
- **Target Metric**: MSE < 3 on test set
- **Model Selection**: Compare multiple regression algorithms
- **Best Model**: Select and save the optimal performing model

## 📈 Dataset Features

### Temporal Features
- `rental_date`: Date and time of DVD rental
- `return_date`: Date and time of DVD return

### Financial Features
- `amount`: Rental payment amount
- `amount_2`: Squared amount
- `rental_rate`: DVD rental rate
- `rental_rate_2`: Squared rental rate
- `replacement_cost`: DVD replacement cost

### Movie Features
- `release_year`: Movie release year
- `length`: Movie duration in minutes
- `length_2`: Squared movie length
- `special_features`: Additional DVD features (trailers, deleted scenes, etc.)

### Rating Features (Pre-encoded Dummies)
- `NC-17`: Adult rating indicator
- `PG`: Parental guidance rating
- `PG-13`: Parental guidance (13+) rating
- `R`: Restricted rating

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and feature engineering
- **scikit-learn**: Regression models and evaluation
- **datetime**: Date calculations for rental duration
- **Model algorithms**: Multiple regression techniques

## 📁 Project Structure
```
predicting-dvd-rental-duration/
├── README.md              # This file
├── data/                  # rental_info.csv dataset
├── notebooks/             # Jupyter notebooks with modeling
├── images/                # Model performance visualizations
└── src/                   # Python modeling scripts
```

## 💡 Key Analysis Steps

### 1. Data Preprocessing
- Load `rental_info.csv` dataset
- Calculate rental duration: `rental_length_days = return_date - rental_date`
- Create dummy variables for special features
- Handle categorical encoding for movie ratings

### 2. Feature Engineering
```python
# Rental duration calculation
df['rental_length_days'] = (df['return_date'] - df['rental_date']).dt.days

# Special features dummies
df['deleted_scenes'] = (df['special_features'] == 'Deleted Scenes').astype(int)
df['behind_the_scenes'] = (df['special_features'] == 'Behind the Scenes').astype(int)
```

### 3. Data Leakage Prevention
Exclude features that contain information about the target:
- `return_date`: Directly reveals rental duration
- Original `special_features`: Replaced by dummy variables

### 4. Model Development
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9
)
```

### 5. Model Evaluation
Test multiple regression algorithms:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## 🎯 Skills Demonstrated

### Data Engineering
- Date-time feature engineering
- Categorical variable encoding
- Dummy variable creation
- Data leakage identification and prevention

### Model Development
- Multiple regression algorithm comparison
- Hyperparameter tuning
- Model evaluation with MSE metric
- Train-test split methodology

### Business Analytics
- Inventory optimization modeling
- Customer behavior prediction
- Performance metric interpretation
- Business requirement translation

## 📊 Deliverables

### Primary Outputs
- `best_model`: Optimal regression model (MSE < 3 on test set)
- `best_mse`: Test set MSE of the best model
- `X_train`, `X_test`: Feature matrices for training and testing
- `y_train`, `y_test`: Target vectors for training and testing

## 🏢 Business Impact

### For DVD Rental Company
- **Inventory Planning**: Predict demand for DVDs based on expected rental duration
- **Resource Allocation**: Optimize purchasing decisions for new inventory
- **Customer Service**: Better stock availability through accurate predictions
- **Operational Efficiency**: Reduce overstock and understock situations

### Key Performance Indicator
- **Target MSE**: ≤ 3 days prediction error
- **Business Value**: Improved inventory turnover and customer satisfaction

## 🔍 Methodology

1. **Data Import and Exploration**: Load and understand dataset structure
2. **Feature Engineering**: Create rental duration and special feature dummies
3. **Data Cleaning**: Handle missing values and data type conversions
4. **Feature Selection**: Identify predictive features, avoid data leakage
5. **Model Training**: Train multiple regression algorithms
6. **Model Evaluation**: Compare MSE scores on test set
7. **Model Selection**: Choose best performing model meeting business requirement

## 📈 Success Criteria
- MSE < 3 on test set
- Model generalizes well to unseen data
- Interpretable features for business stakeholders
- Reproducible results with random_state=9

## 🔗 Related Projects
- [Modeling Car Insurance Claim Outcomes](../modeling-car-insurance-claim-outcomes/)
- [Customer Analytics Data Modeling](../customer-analytics-data-modeling/)

## 📄 Data Source
DVD rental company transaction records containing customer rental patterns, movie characteristics, and temporal information.

---

**Project Status**: ✅ Completed  
**Last Updated**: [Date]  
**Business Focus**: Inventory optimization through rental duration prediction