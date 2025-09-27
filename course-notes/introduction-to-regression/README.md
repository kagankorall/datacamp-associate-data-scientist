# Introduction to Regression

## Course Overview
This course covers the fundamentals of regression analysis, including linear regression, multiple regression, model evaluation, and interpretation of results.

## Key Topics Covered

### 1. Linear Regression Fundamentals
- Simple linear regression
- Assumptions of linear regression
- Least squares method
- Interpreting coefficients

### 2. Multiple Regression
- Multiple linear regression
- Feature selection
- Multicollinearity
- Interaction effects

### 3. Model Evaluation
- R-squared and adjusted R-squared
- Residual analysis
- Cross-validation
- Model diagnostics

### 4. Advanced Topics
- Polynomial regression
- Regularization (Ridge, Lasso)
- Logistic regression basics
- Non-linear relationships

## Key Concepts

### Simple Linear Regression
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats

# Simple linear regression
def simple_linear_regression(X, y):
    """Perform simple linear regression."""
    
    # Reshape X if it's 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Create and fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'intercept': model.intercept_,
        'coefficient': model.coef_[0]
    }

# Example usage
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 14.0, 15.8, 18.1, 20.0])

results = simple_linear_regression(X, y)
print(f"R-squared: {results['r2']:.4f}")
print(f"Intercept: {results['intercept']:.4f}")
print(f"Coefficient: {results['coefficient']:.4f}")
```

### Multiple Linear Regression
```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

def multiple_linear_regression(X, y, feature_names=None):
    """Perform multiple linear regression with feature analysis."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Feature importance
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'feature_importance': feature_importance,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }
```

### Model Diagnostics and Residual Analysis
```python
def plot_regression_diagnostics(y_true, y_pred, residuals=None):
    """Create comprehensive regression diagnostic plots."""
    
    if residuals is None:
        residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    
    # 3. Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    
    # 4. Histogram of residuals
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print("Diagnostic Tests:")
    print(f"Mean of residuals: {np.mean(residuals):.6f}")
    print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test p-value: {shapiro_p:.4f}")
    print("Residuals are normally distributed" if shapiro_p > 0.05 else "Residuals are not normally distributed")

def check_multicollinearity(X, feature_names=None):
    """Check for multicollinearity using VIF."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    print("Variance Inflation Factors:")
    print(vif_data.sort_values('VIF', ascending=False))
    print("\nVIF > 10 indicates high multicollinearity")
    
    return vif_data
```

### Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def polynomial_regression(X, y, degree=2):
    """Perform polynomial regression."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create polynomial features and model pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    # Fit model
    poly_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = poly_pipeline.predict(X_train)
    y_test_pred = poly_pipeline.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return {
        'pipeline': poly_pipeline,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }

# Compare different polynomial degrees
def compare_polynomial_degrees(X, y, max_degree=5):
    """Compare polynomial regression with different degrees."""
    
    results = []
    for degree in range(1, max_degree + 1):
        result = polynomial_regression(X, y, degree=degree)
        results.append({
            'degree': degree,
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2']
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_df['degree'], comparison_df['train_r2'], 
             'o-', label='Train R²', linewidth=2)
    plt.plot(comparison_df['degree'], comparison_df['test_r2'], 
             's-', label='Test R²', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('Model Performance vs Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return comparison_df
```

### Regularized Regression
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

def regularized_regression(X, y, alpha_range=None):
    """Compare Ridge, Lasso, and ElasticNet regression."""
    
    if alpha_range is None:
        alpha_range = np.logspace(-4, 2, 20)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to compare
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=1000),
        'ElasticNet': ElasticNet(max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        # Grid search for best alpha
        param_grid = {'alpha': alpha_range}
        if name == 'ElasticNet':
            param_grid['l1_ratio'] = [0.1, 0.5, 0.7, 0.9]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        
        results[name] = {
            'best_params': grid_search.best_params_,
            'train_score': grid_search.best_score_,
            'test_score': r2_score(y_test, y_pred),
            'model': best_model
        }
    
    # Compare results
    comparison = pd.DataFrame({
        name: {
            'Best Alpha': result['best_params']['alpha'],
            'CV Score': result['train_score'],
            'Test Score': result['test_score']
        } for name, result in results.items()
    }).T
    
    print("Regularized Regression Comparison:")
    print(comparison)
    
    return results, comparison
```

### Cross-Validation and Learning Curves
```python
from sklearn.model_selection import cross_val_score, learning_curve

def plot_learning_curves(model, X, y, cv=5):
    """Plot learning curves to diagnose bias/variance."""
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, 's-', label='Validation Score', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return train_sizes, train_scores, val_scores

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation and return detailed results."""
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    print(f"Cross-Validation Results:")
    print(f"Mean R² Score: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    print(f"Individual Scores: {cv_scores}")
    
    return cv_scores
```

## Course Notes

# Simple Linear Regression Modeling

Regression is a statistical model to explore the relationship between a response variable and some explanatory variables. Given values of explanatory variables, you can predict the values of the response variable.

Response variable (dependent variable): The variable that you want to predict

Explanatory variable (independent variable): The variables that explain how the response variable will change

In Linear Regression, the response variable is numeric.

In Logistic Regression, the response variable is logical.

```python
# Example
# Import seaborn with alias sns
import seaborn as sns

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Draw the scatter plot
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=taiwan_real_estate)

# Draw a trend line on the scatter plot of price_twd_msq vs. n_convenience
sns.regplot(x='n_convenience',
         y='price_twd_msq',
         data=taiwan_real_estate,
         ci=None,
         scatter_kws={'alpha': 0.5})

# Show the plot
plt.show()
```

## Fitting a Linear Regression

Intercept: The y value at the point when x is zero

Slope: The amount the y value increase if you increase x by one amount 

Equation: y = intercept + (slope * x)

```python
from statsmodels.formula.api import ols
mdl_payment_vs_claims = ols('total_payment_sek tilda n_claims',
															data = swedish_motor_insurance)
mdl_payment_vs_claims = mdl_payment_vs_claims.fit()
print(mdl_payment_vs_claims.params)
```

## Categorical explanatory variables

```python
# Create the model, fit it
mdl_price_vs_age = ols('price_twd_msq ~ house_age_years', data=taiwan_real_estate).fit()

# Print the parameters of the fitted model
print(mdl_price_vs_age.params)
```

# Predictions and Model Objects

## Making Predictions

```python
# Import numpy with alias np
import numpy as np

# Create explanatory_data 
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})

# Use mdl_price_vs_conv to predict with explanatory_data, call it price_twd_msq
price_twd_msq = mdl_price_vs_conv.predict(explanatory_data)

# Print it
print(price_twd_msq)
```

## Working with Model Objects

.fittedvalues attribute: Fitted values are predictions on the original dataset

.resid attribute: Residuals are actual response values minus predicted response values

.summary attribute: Summary shows the detail of the model

## Regression to the mean

Response value = fitted value + residual

The stuff you explained + The stuff you couldn’t explain

Residuals exist due to problems in the model and fundamental randomness

Extreme cases are often due to randomness

Regression to the mean means extreme cases don’t persist over time

```python
# Example
# Create a new figure, fig
fig = plt.figure()

# Plot the first layer: y = x
plt.axline(xy1=(0,0), slope=1, linewidth=2, color="green")

# Add scatter plot with linear regression trend line
sns.regplot(data = sp500_yearly_returns, x='return_2018', y='return_2019', ci= None)

# Set the axes so that the distances along the x and y axes look the same
plt.axis('equal')

# Show the plot
plt.show()
```

## Transforming Variables

```python
# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])

plt.figure()

# Plot using the transformed variable
sns.regplot(x='sqrt_dist_to_mrt_m', y='price_twd_msq', data=taiwan_real_estate)
plt.show()
```

# Assessing Model Fit

## Quantifying model fit

In this chapter, we will see quality of our models by using some methods.

### R-Squared

R-squared is called coefficient of determination. The proportion of the variance in the response variable that is predictable from the explanatory variable.

1 means a perfect fit. 0 means the worst possible fit.

```python
print(mdl_bream.rsquared)
```

R-squared is just correlation squared.

```python
coeff_determination = bream['length_cm'].corr(bream['mass_g']) ** 2
```

### Residual Standard Error (RSE)

A typical difference between a prediction and an observed response. It has the same unit as the response variable.

MSE = RSE2

## Visualizing Model Fit

Residual properties of a good fit should be:

- Residuals are normally distributed
- The mean of the residuals is zero

## Outliers, Leverage and Influence

Leverage is a measure of how extreme the explanatory variable values are.

Influence measures how much the model would change if you left the observation out of the dataset when modeling.

```python
# Create summary_info
summary_info = mdl_price_vs_dist.get_influence().summary_frame()

# Add the hat_diag column to taiwan_real_estate, name it leverage
taiwan_real_estate["leverage"] = summary_info["hat_diag"]

# Add the cooks_d column to taiwan_real_estate, name it cooks_dist
taiwan_real_estate['cooks_dist'] = summary_info['cooks_d']

# Sort taiwan_real_estate by cooks_dist in descending order and print the head.
print(taiwan_real_estate.sort_values('cooks_dist', ascending=False).head())
```

# Simple Logistic Regression Modeling

## Logistic Regression

It is another type of generalized linear model. 

Used when the response variable is logical.

The responses follow logistic (S-shaped) curve.

![image.png](attachment:15805159-90f3-4b28-b4db-abbbc0caab58:image.png)

```python
# Example
# Draw a linear regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn, 
            ci=None,
            line_kws={"color": "red"})

# Draw a logistic regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x='time_since_first_purchase', y='has_churned', data=churn, ci=None, logistic=True, line_kws={'color':'blue'})

plt.show()
```

## Predictions and odds ratios

Odds Ratio: It is the probability of something happening divided by the probability that it doesn’t.

$$
oddsratio = probability / (1-probability)
$$

```python
# Create prediction_data
prediction_data = explanatory_data.assign(
    has_churned = mdl_churn_vs_relationship.predict(explanatory_data)
)

fig = plt.figure()

# Create a scatter plot with logistic trend line
sns.regplot(x='time_since_first_purchase', y='has_churned', data=churn, logistic=True)

# Overlay with prediction_data, colored red
sns.scatterplot(x='time_since_first_purchase', y='has_churned', data=prediction_data)

plt.show()
```