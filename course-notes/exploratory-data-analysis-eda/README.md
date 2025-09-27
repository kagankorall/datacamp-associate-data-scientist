# Exploratory Data Analysis (EDA)

## Course Overview
This course covers the fundamentals of exploratory data analysis, including data inspection, visualization techniques, and identifying patterns in datasets.

## Key Topics Covered

### 1. Data Inspection
- Initial data exploration
- Data types and structure
- Missing values detection
- Outlier identification

### 2. Descriptive Statistics
- Summary statistics
- Distribution analysis
- Central tendency measures
- Variability measures

### 3. Data Visualization
- Histograms and box plots
- Scatter plots and correlation analysis
- Bar charts and categorical data
- Time series visualization

### 4. Pattern Recognition
- Trends and seasonality
- Relationships between variables
- Data quality assessment

## Key Concepts

### Initial Data Exploration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and inspect data
df = pd.read_csv('data.csv')

# Basic information
print(df.info())
print(df.describe())
print(df.head())
print(df.shape)

# Check for missing values
print(df.isnull().sum())

# Data types
print(df.dtypes)
```

### Descriptive Statistics
```python
# Summary statistics
print(df.describe(include='all'))

# Specific statistics
print(f"Mean: {df['column'].mean()}")
print(f"Median: {df['column'].median()}")
print(f"Standard Deviation: {df['column'].std()}")

# Quantiles
print(df['column'].quantile([0.25, 0.5, 0.75]))
```

### Visualization Techniques
```python
# Distribution analysis
plt.figure(figsize=(12, 8))

# Histogram
plt.subplot(2, 2, 1)
plt.hist(df['numerical_column'], bins=30, alpha=0.7)
plt.title('Distribution of Numerical Column')

# Box plot
plt.subplot(2, 2, 2)
plt.boxplot(df['numerical_column'])
plt.title('Box Plot')

# Scatter plot
plt.subplot(2, 2, 3)
plt.scatter(df['x_column'], df['y_column'], alpha=0.6)
plt.xlabel('X Column')
plt.ylabel('Y Column')
plt.title('Scatter Plot')

# Correlation heatmap
plt.subplot(2, 2, 4)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()
```

### Categorical Data Analysis
```python
# Value counts
print(df['category_column'].value_counts())

# Bar plot
plt.figure(figsize=(10, 6))
df['category_column'].value_counts().plot(kind='bar')
plt.title('Distribution of Categories')
plt.xticks(rotation=45)
plt.show()

# Cross-tabulation
cross_tab = pd.crosstab(df['category1'], df['category2'])
print(cross_tab)
```

### Outlier Detection
```python
# Using IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['column']))
outliers_zscore = df[z_scores > 3]
```

## Course Notes

```python
books = pd.read_csv('books.csv')

# Counting number of values in categorical data
books.value_counts("genre")

# Numerical columns description
books.describe()

# Summary of column non-missing values and data types
books.info()
```

```python
# Plotting histogram
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=books, x="rating")
plt.show()
```

## Data Validation

```python
# For example in a dataset year column is float and we want it as whole number
books["year"] = books["year"].astype(int)

books["genre"].isin(["Fiction","Non Fiction"])

# For example, we want to see min and max years in out dataset
books["year"].min()

books["year"].max()
```

```python
# Define a Series describing whether each continent is outside of Oceania
not_oceania = ~unemployment["continent"].isin(["Oceania"])

# Print unemployment without records related to countries in Oceania
print(unemployment[not_oceania])
```

```python
# Print the minimum and maximum unemployment rates during 2021
print(unemployment["2021"].min(), unemployment["2021"].max())

# Create a boxplot of 2021 unemployment rates, broken down by continent
sns.boxplot(data = unemployment, x="2021", y="continent")
plt.show()
```

## Data Summarization

.groupby() groups data by category.

Aggregating function indicated how to summarize grouped data

```python
books[["genre", "rating", "year"]].groupby("genre").mean()
```

Output:

![image.png](attachment:25d6c341-ec98-4361-9180-32daa6929ce8:b653558f-a50f-489a-8ca1-7fc358f88112.png)

.agg() → Applies aggregating functions across a DataFrame

```python
books[["rating", "year"]].agg(["mean", "std"])
```

Output:

![image.png](attachment:f5ecf5d6-b342-471f-8be4-53f93129ef27:image.png)

```python
books.agg({"rating": ["mean", "std"], "year":["median"]})
```

```python
books.groupby("genre").agg(
		mean_rating = ("rating", "mean"),
		std_rating = ("rating", "std"),
		median_year = ("year", "median")
)
```

![image.png](attachment:41c04307-f90f-405f-a0c7-ad3730274fb4:image.png)

```python
sns.barplot(data=books, x="genre", y="rating")
plt.show()
```

```python
# Print mean and standard deviation grouped by continent
print(unemployment[["continent", "2019", "2020"]].groupby("continent").agg(["mean","std"]))
```

## Data Cleaning and Imputation

### Addressing missing data

```python
# Showing number of missing values in each column in a dataset
print(salaries.isna().sum())
```

**Strategies for Addressing Missing Data**:

**1. Deletion Strategy**:

- **Drop missing values** when they represent 5% or less of the total dataset
- This approach is simple and maintains data integrity but reduces sample size

**2. Imputation by Central Tendency**:

- **Mean imputation**: Use for normally distributed continuous variables
- **Median imputation**: Use for skewed distributions or when outliers are present
- **Mode imputation**: Use for categorical variables
- Choice depends on the variable's distribution and the context of the analysis

**3. Group-Based Imputation**:

- **Impute by sub-groups** when logical groupings exist in the data
- Example: Different experience levels may have different median salaries, so impute missing salary values based on the median salary within each experience level group
- This approach preserves relationships between variables and provides more accurate estimates

```python
threshold = len(salaries) * 0.05

cols_to_drop = salaries.columns[salaries.isna().sum() <= threshold]

salaries.dropna(subset = cols_to_drop, inplace = True)

cols_with_missing_values = salaries.columns[salaries.isna().sum() > 0]

for col in cols_with_missing_values[:-1]:
		salaries[col].fillna(salaries[col].mode()[0])
```

```python
# Imputing by sub-group
salaries_dict = salaries.groupby("Experience")["Salary_USD"].median().to_dict()

salaries["Salary_USD"] = salaries["Salary_USD"].fillna(salaries["Experience"].map(salaries_dict))
```

```python
# Example
# Count the number of missing values in each column
print(planes.isna().sum())

# Find the five percent threshold
threshold = len(planes) * 0.05

# Create a filter
cols_to_drop = planes.columns[planes.isna().sum() <= threshold]

# Drop missing values for columns below the threshold
planes.dropna(subset=cols_to_drop, inplace=True)

print(planes.isna().sum())

```

### Converting and Analyzing Categorical Data

```python
# Counting unique job titles
print(salaries["Designation"].nunique())
```

pandas.Series.str.contains() → Search a column for a specific string or multiple strings

```python
salaries["Designation"].str.contains("Machine Learning|AI")

# Words of interest: Any that starts with Data
salaries["Designation"].str.contains("^Data")
```

```python
 # Example
 # Filter the DataFrame for object columns
non_numeric = planes.select_dtypes("object")

# Loop through columns
for col in non_numeric.columns:
  
  # Print the number of unique values
  print(f"Number of unique values in {col} column: ", non_numeric[col].nunique())
  
# Create a list of categories
flight_categories = ["Short-haul", "Medium", "Long-haul"]

# Create short-haul values
short_flights = "^0h|^1h|^2h|^3h|^4h"

# Create medium-haul values
medium_flights = "^5h|^6h|^7h|^8h|^9h"

# Create long-haul values
long_flights = "^10h|^11h|^12h|^13h|^14h|^15h|^16h"

# Create conditions for values in flight_categories to be created
conditions = [
    (planes["Duration"].str.contains(short_flights)),
    (planes["Duration"].str.contains(medium_flights)),
    (planes["Duration"].str.contains(long_flights))
]

# Apply the conditions list to the flight_categories
planes["Duration_Category"] = np.select(conditions, 
                                        flight_categories,
                                        default="Extreme duration")

# Plot the counts of each category
sns.countplot(data=planes, x="Duration_Category")
plt.show()
```

### Working with Numeric Data

```python
# Converting strings to numbers
salaries["Salary_In_Rupees"] = salaries["Salary_In_Rupees"].str.replace(",","")

salaries["Salary_In_Rupees"] = salaries["Salary_In_Rupees"].astype(float)
```

```python
# Adding summary statistics into a dataframe
salaries.groupby("Company_Size")["Salary_USD"].mean()

salaries["std_dev"] = salaries.groupby("Experience")["Salary_USD"].transform(lambda x: x.std())

print(salaries[["Experience", "std_dev"]].value_counts())
```

Output:

![image.png](attachment:e7bd395b-b840-406a-ad11-72ddf7063511:image.png)

### Handling Outliers

An outlier is an observation far away from other data points.

![image.png](attachment:90c78eac-9a04-48ea-97bc-75bd0eb30c5e:image.png)

Upper outliers → 75th percentile + (1.5 * IQR)

Lower outliers → 25th percentile - (1.5 * IQR)

```python
seventy_fifth = salaries["Salary_USD"].quantile(0.75)
twenty_fifth = salaries["Salary_USD"].quantile(0.25)

salaries_iqr = seventy_fifth - twenty_fifth

#Upper threshold
upper = seventy_fifth + (salaries_iqr * 1.5)

# Lower threshold
lower = twenty_fifth - (salaries_iqr * 1.5)

no_outliers = salaries[(salaries["Salary_USD"] > lower) & (salaries["Salary_USD"] < upper)]
```

## Relationships in Data

DateTime data needs to be explicitly declared to Pandas.

```python
# Converting string to a datetime value
divorce = pd.read_csv("divorce.csv", parse_dates = ["marriage_date"])

divorce["marriage_date"] = pd.to_datetime(divorce["marriage_date"])

# Creating DateTime data
divorce["marriage_date"] = pd.to_datetime(divorce[["month", "day", "year"]])

# Extract parts of a full data
divorce["marriage_month"] = divorce["marriage_date"].dt.month
divorce["marriage_year"] = divorce["marriage_date"].dt.year
divorce["marriage_day"] = divorce["marriage_date"].dt.day

# Example
# Define the marriage_year column
divorce["marriage_year"] = divorce["marriage_date"].dt.year

# Create a line plot showing the average number of kids by year
sns.lineplot(data = divorce, x="marriage_year", y="num_kids")
plt.show()
```

### Correlation

It describes direction and strength of relationship between two variables.

```python
sns.heatmap(divorce.corr(numeric_only = True), annot = True)
plt.show()
```

### Kernel Density Estimate (KDE)

```python
sns.kdeplot(data=divorce, x="marriage_duration",hue="education_man", cut = 0, cumulative=True)
plt.show()
```

## Actions in EDA

We perform EDA for detecting patterns and relationships. Also, for generating questions or hypotheses. Preparing daha for machine learning.

```python
planes["Destination"].value_counts(normalize=True)
```

```python
# Cross tabulation
pd.crosstab(planes["Source"],planes["Destination"], values=planes["Price"], aggfunc="median")
```

### Generating new features

```python
# From object to int
planes["Total_Stops"] = planes["Total_Stops"].str.replace(" stops","")
planes["Total_Stops"] = planes["Total_Stops"].str.replace(" stop","")
planes["Total_Stops"] = planes["Total_Stops"].str.replace("non-stop","0")
planes["Total_Stops"] = planes["Total_Stops"].astype(int)
```

```python
# Creating categorical data
twenty_fifth = planes["Price"].quantile(0.25)
median = planes["Price"].median()
seventy_fifth = planes["Price"].quantile(0.75)
maximum = planes["Price"].max()

labels = ["Economy", "Premium Economy", "Business Class", "First Class"]
bins = [0, twenty_fifth, median, seventy_fifth, maximum]

planes["Price_Category"] = pd.cut(planes["Price"],
																	labels=labels,
																	bins=bins)
```

### Generating Hypotheses

For detecting relationships, differences, and patterns we use Hypothesis Testing. Hypothesis testing requires, prior to data collection:

- Generating a hypothesis or question
- A decision on what statistical test to use

```python
# Filter for employees in the US or GB
usa_and_gb = salaries[salaries["Employee_Location"].isin(["US", "GB"])]

# Create a barplot of salaries by location
sns.barplot(data=usa_and_gb, x="Employee_Location", y="Salary_USD")
plt.show()
```