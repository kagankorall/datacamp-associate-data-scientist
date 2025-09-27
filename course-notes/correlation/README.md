# Correlation

## Course Overview
This course covers correlation analysis, measuring relationships between variables, and interpreting correlation coefficients.

## Key Topics Covered

### 1. Correlation Fundamentals
- What is correlation?
- Types of correlation
- Correlation vs causation

### 2. Correlation Coefficients
- Pearson correlation coefficient
- Spearman rank correlation
- Kendall's tau

### 3. Visualization and Interpretation
- Scatter plots
- Correlation matrices
- Heatmaps

## Key Concepts

### Pearson Correlation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate Pearson correlation
correlation = df['variable1'].corr(df['variable2'])
print(f"Correlation: {correlation}")
```

### Correlation Matrix
```python
# Create correlation matrix
corr_matrix = df.corr()

# Visualize with heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

### Scatter Plot with Correlation
```python
# Scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='variable1', y='variable2')
sns.regplot(data=df, x='variable1', y='variable2', scatter=False)
plt.title('Relationship between Variable 1 and Variable 2')
plt.show()
```

## Course Notes

## **Correlation**

**Correlation Coefficient**: A statistical measure that quantifies the strength and direction of the linear relationship between two variables. The correlation coefficient is always a number between -1 and 1.

**Interpretation**:

- **Magnitude**: The absolute value indicates the strength of the relationship
    - Values close to 1 or -1 indicate a strong linear relationship
    - Values close to 0 indicate a weak or no linear relationship
- **Sign**: The positive or negative sign indicates the direction of the relationship
    - **Positive correlation (+)**: As one variable increases, the other variable also increases
    - **Negative correlation (-)**: As one variable increases, the other variable decreases

**Key Points**:

- A correlation of +1 indicates a perfect positive linear relationship
- A correlation of -1 indicates a perfect negative linear relationship
- A correlation of 0 indicates no linear relationship between the variables
- Correlation measures only linear relationships and may miss non-linear associations
- If x is correlated with y, it does not mean x causes y (Confounding)

```python
import seaborn as sns
sns.scatterplot(x='sleep_total', y = 'sleep_rem', data = msleep)
plt.show()
# Adding trendline
sns.lmplot(x='sleep_total', y = 'sleep_rem', data = msleep, ci= None)
plt.show()
# Calculating correlation
msleep['sleep_total'].corr(msleep['sleep_rem'])
```

![image.png](attachment:36a1c257-48f9-4cd1-83da-232859f55435:image.png)

To make relationship more linear we can use these transformations:

- Log transformation (log(x))
- Square root transformation (ssqrt(x))
- Reciprocal transformation (1/x)

Certain statistical methods rely on variables having a linear relationships

- Correlation coefficient
- Linear regression