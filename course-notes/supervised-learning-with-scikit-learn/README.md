# Supervised Learning with Scikit-Learn

## Course Overview
This course covers supervised learning techniques using scikit-learn, including classification and regression algorithms, model evaluation, and hyperparameter tuning.

## Key Topics Covered

### 1. Classification Algorithms
- k-Nearest Neighbors (kNN)
- Logistic Regression
- Decision Trees
- Support Vector Machines

### 2. Regression Algorithms
- Linear Regression
- Ridge and Lasso Regression
- Decision Tree Regression
- Random Forest

### 3. Model Evaluation
- Train-test split
- Cross-validation
- Performance metrics
- Confusion matrices

### 4. Model Selection and Tuning
- Grid search
- Random search
- Pipeline creation
- Feature preprocessing

## Key Concepts

### Basic Classification Example
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(f"kNN Accuracy: {accuracy_score(y_test, knn_pred):.3f}")

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred):.3f}")
```

### Model Evaluation
```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Cross-validation
cv_scores = cross_val_score(knn, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Confusion Matrix
cm = confusion_matrix(y_test, knn_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print(classification_report(y_test, knn_pred))
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Grid search
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Regression Example
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Multiple regression models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Compare models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE={mse:.3f}, R²={r2:.3f}")
```

### Feature Preprocessing Pipeline
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Preprocessing for different column types
numeric_features = ['age', 'income']
categorical_features = ['category', 'region']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)  # or use OneHotEncoder
    ])

# Complete pipeline
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit and evaluate
clf_pipeline.fit(X_train, y_train)
accuracy = clf_pipeline.score(X_test, y_test)
print(f"Pipeline Accuracy: {accuracy:.3f}")
```

## Course Notes

# Classification

## Machine Learning with scikit-learn

The predicted values are known in supervised learning. In Supervised learning, aim is predict the target values of unseen data, given the features. Classification and regression are supervised learning methods. In classification, target variable consists of categories. In regression, target variable is continuous.

Feature = predictor variable = independent variable

Target variable = dependent variable = response variable

## The Classification Challenge

### k-Nearest Neighbors (kNN)

Predict the label of a data point by,

- Looking at the k closest labeled data points
- Taking a majority vote

```python
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[['total_day_charge', 'total_eve_charge']].values
y = churn_df['churn'].values

knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X,y)

predictions = knn.predict(X_new)
```

## Measuring Model Performance

In classification, accuracy is a commonly used metric.

Accuracy = correct predictions / total observations

![image.png](attachment:87d36df5-8e2c-4c84-99e8-4b91089a8b09:image.png)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
																		test_size = 0.3, random_state = 21,
																		stratify = y)
```

In kNN, larger k means less complex model which means it can cause underfitting. Smaller k means more complex model which means it can lead to overfitting.

```python
# kNN Example
# Import the module
from sklearn.model_selection import train_test_split

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
```

# Regression

## Introduction to Regression

```python
X = diabetes_df.drop('glucose', axis = 1).values
y = diabetes_df['glucose'].values

# Making predictions from a single feature
X_bmi = X[:,3]
X_bmi = X_bmi.reshape(-1, 1)

# Fitting a regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

# Plotting
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.show()
```

## The Basics of Linear Regression

y = ax + b

In this equation, y is target, x is single feature and a,b are parameters/coefficients of the model - slope, intercept.

So how do we choose a and b? First, define an error function for any given line. Then, choose the line that minimizes the error function. Error function = loss function = cost function

The distance between line and observation called residual. 

### Ordinary Least Squares

![image.png](attachment:f9f34261-620b-4d63-87d3-905e8a282cd0:image.png)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
```

### R-squared

R^2: quantifies the variance in target values explained by the features. It ranges from 0 to 1.

```python
reg_all.score(X_test, y_test)
```

### Mean Squared Error & Root Mean Squared Error

![image.png](attachment:c80795bd-76de-46e2-85a7-96c97a703148:image.png)

```python
from sklearn.metrics import root_mean_squared_error
root_mean_squared_error(y_test, y_pred)
```

```python
# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

# Import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = root_mean_squared_error(y_test, y_pred)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
```

## Cross-Validation

![image.png](attachment:13820f9a-e9ae-463f-83fa-0ff30ae24f02:image.png)

5 folds = 5-fold CV

More folds mean more computationally expensive

```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)
```

```python
# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)
```

## Regularized Regression

To minimize overfitting

![image.png](attachment:306f20d8-1c27-4961-a3a2-487032d84d98:image.png)

```python
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
		ridge = Ridge(alpha = alpha)
		ridge.fit(X_train, y_train)
		y_pred = ridge.predict(X_test)
		scores.append(ridge.score(X_test, y_test))
print(scores)
```

![image.png](attachment:f333f09a-bfe1-4e37-9140-d0d9fe292826:image.png)

```python
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
	lasso = Lasso(alpha = alpha)
	lasso.fit(X_train, y_train)
	lasso_pred = lasso.predict(X_test)
	scores.append(lasso.score(X_test, y_test))
print(scores)
```

Lasso can select important features of a dataset.

Shrinks the coefficients of less important features to zero.

Features not shrunk to zero are selected by lasso.

# Fine-Tuning Your Model

## How good is your model?

### Class Imbalance

Classification for predicting fraudulent bank transactions. For example, 99% of transactions are legit but 1% are fraudulent. Therefore, imbalance class will affect model accuracy.

### Confusion Matrix

![image.png](attachment:75fcc0ed-5621-46b5-8b4f-4d5a158c521a:image.png)

Precision: tp / (tp + fp)

High precision means lower false positive rate. For our example, high precision means not many legitimate transactions are predicted to be fraudulent. 

Recall: tp / (tp + fn)

High recall means lower false positive rate. For our example, high recall means predicted most fraudulent transactions correctly. 

F1 Score: 2 * ((precision * recall) / (precision + recall))

```python
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

confusion_matrix(y_test, y_pred)

classification_report(y_test, y_pred)
```

## Logistic Regression and the ROC Curve

Logistic Regression is used for classification problems. It outputs probabilities. If the probability (p) > 0.5 then the data is labeled as 1.

```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

y_pred_probs = logreg.predict_proba(X_test)[:,1]
```

For logistic regression default threshold value is 0.5. 

![image.png](attachment:8023a01f-154d-4085-84ce-7371c17f29a1:image.png)

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

AUC → Area under curve (roc curve)

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

## Hyperparameter Tuning

For ridge/lasso regression: Choosing alpha

KNN: Choosing n_neighbors

It is essential to use cross-validation to avoid overfitting to the test set.

```python
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": np.arange(0.0001, 1, 10),
							"solver": ["sag", "lsqr]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv = kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)

```

### Limitations and an alternative approach

3 fold cross validation, 1 hyperparameter, 10 total values = 30 fits

RandomizedSearchCV → Alternative approach

![image.png](attachment:c6f4abaf-d885-47df-81ac-33b16a984d40:image.png)

# Preprocessing and Pipelines

## Preprocessing Data

Convert categorical values to dummy variables before putting them into model.

In scikit-learn: OneHotEncoder()

In pandas: get_dummies()

```python
music_df = pd.read_csv('music.csv')
music_dummies = pd.get_dummies(music_df['genre'], drop_first = True)

music_dummies = pd.concat([music_df, music_dummies], axis = 1)
music_dummies = music_dummies.drop('genre', axis = 1)
```

## Handling Missing Data

Means no value for a feature in a particular row.

```python
music_df.isna().sum().sort_values()
```

### Removing Null Values

First method is dropping null values in dataset. If these data is no more than 5% of the data.

```python
music_df = music_df.dropna(subset = ['genre','popularity','loudness','liveness','tempo'])
```

### Imputing Values

Imputation - use subject-matter expertise to replace missing data with educated guesses.

Common to use the mean. We can also use the median or another value. For categorical values, we typically use the most frequent value (the mode). 

But first we must split our data first, to avoid data leakage.

```python
from sklearn.impute import SimpleImputer
X_cat = music_df['genre'].values.reshape(-1, 1)
X_num = music_df.drop(['genre','popularity'], axis = 1).values
y = music_df['popularity'].values
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=12)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=12)
imp_cat = SimpleImputer(strategy = 'most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)

imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis = 1)
```

```python
# Imputing within a pipeline
from sklearn.pipeline import Pipeline

music_df = music_df.dropna(subset = ['genre','popularity','loudness','liveness','tempo'])
music_df['genre'] = np.where(music_df['genre'] == 'Rock',1,0)
X = music_df.drop('genre', axis = 1).values
y = music_df['genre'].values

steps = [('imputation', SimpleImputer()),
					('logistic_regression'), LogisticRegression())]

pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

## Centering and Scaling

Why we scale our data? Because first of all, many models use some form of distance to inform them. Features on larger scales can disproportionately influence the model. To solve this we are using normalizing or standardizing methods. 

### How to scale our data?

- Subtract the mean and divide by variance
All features are centered around zero and have a variance of one
This is called standardization
- Can also subtract the minimum and divide by the range
Minimum zero and maximum one
- Can also normalize so the data ranges from -1 to +1

```python
from sklearn.preprocessing import StandardScaler
X = music_df.drop('genre', axis = 1).values
y = music_df['genre'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
# Scaling in Pipeline
steps = [('scaler', StandardScaler()),
				 ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
```

```python
# Scaling + CV in a pipeline
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler()),
				 ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1,50)}

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
```

```python
# Example
# Build the steps
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=21)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)
```

## Evaluating Multiple Models

Regression model performance:

- RMSE
- R-squared

Classification model performance:

- Accuracy
- Confusion Matrix
- Precision, Recall, F1 Score
- ROC AUC