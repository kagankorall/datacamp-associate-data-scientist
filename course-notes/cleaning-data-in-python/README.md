# Cleaning Data in Python

## Course Overview
This course covers essential data cleaning techniques in Python, including handling missing values, removing duplicates, data type conversions, and outlier detection.

## Key Topics Covered

### 1. Missing Data Handling
- Identifying missing values
- Strategies for missing data
- Imputation techniques
- Dropping vs filling missing values

### 2. Duplicate Detection and Removal
- Finding duplicate records
- Different types of duplicates
- Removing duplicates strategically

### 3. Data Type Conversions
- Converting between data types
- Handling inconsistent formats
- Date and time conversions
- String manipulations

### 4. Outlier Detection and Treatment
- Statistical methods for outlier detection
- Visualization techniques
- Outlier treatment strategies

## Key Concepts

### Missing Data Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check for missing values
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # Percentage

# Visualize missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Missing data patterns
import missingno as msno
msno.matrix(df)
msno.bar(df)
```

### Handling Missing Values
```python
# Drop missing values
df_dropped = df.dropna()  # Drop all rows with any missing values
df_dropped = df.dropna(subset=['column1', 'column2'])  # Drop specific columns
df_dropped = df.dropna(thresh=5)  # Keep rows with at least 5 non-null values

# Fill missing values
df['column'].fillna(df['column'].mean(), inplace=True)  # Mean imputation
df['column'].fillna(df['column'].median(), inplace=True)  # Median imputation
df['column'].fillna(df['column'].mode()[0], inplace=True)  # Mode imputation
df['column'].fillna(method='ffill', inplace=True)  # Forward fill
df['column'].fillna(method='bfill', inplace=True)  # Backward fill

# Advanced imputation
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='mean')
df[['col1', 'col2']] = imputer.fit_transform(df[['col1', 'col2']])

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
```

### Duplicate Detection and Removal
```python
# Check for duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")
print(f"Duplicate rows:\n{df[df.duplicated()]}")

# Find duplicates based on specific columns
duplicates = df.duplicated(subset=['column1', 'column2'])
print(f"Duplicates based on columns: {duplicates.sum()}")

# Remove duplicates
df_no_duplicates = df.drop_duplicates()
df_no_duplicates = df.drop_duplicates(subset=['column1', 'column2'])
df_no_duplicates = df.drop_duplicates(keep='first')  # Keep first occurrence
df_no_duplicates = df.drop_duplicates(keep='last')   # Keep last occurrence
```

### Data Type Conversions
```python
# Check current data types
print(df.dtypes)

# Convert data types
df['column'] = df['column'].astype('int64')
df['column'] = df['column'].astype('float64')
df['column'] = df['column'].astype('category')

# Convert to datetime
df['date_column'] = pd.to_datetime(df['date_column'])
df['date_column'] = pd.to_datetime(df['date_column'], format='%Y-%m-%d')

# Handle errors in conversion
df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')

# String cleaning and conversion
df['text_column'] = df['text_column'].str.strip()  # Remove whitespace
df['text_column'] = df['text_column'].str.lower()  # Convert to lowercase
df['text_column'] = df['text_column'].str.replace('[^a-zA-Z0-9]', '', regex=True)  # Remove special characters
```

### Outlier Detection
```python
# Statistical methods
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Z-score method
from scipy import stats
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores > threshold]

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.boxplot(df['column'])
plt.title('Box Plot')

plt.subplot(1, 3, 2)
plt.hist(df['column'], bins=50)
plt.title('Histogram')

plt.subplot(1, 3, 3)
plt.scatter(range(len(df)), df['column'])
plt.title('Scatter Plot')

plt.tight_layout()
plt.show()
```

### Text Data Cleaning
```python
import re

def clean_text(text):
    """Comprehensive text cleaning function."""
    if pd.isna(text):
        return text
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove numbers (optional)
    # text = re.sub(r'\d+', '', text)
    
    return text

# Apply cleaning
df['cleaned_text'] = df['text_column'].apply(clean_text)

# String methods for cleaning
df['text_column'] = (df['text_column']
                     .str.strip()                    # Remove leading/trailing whitespace
                     .str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces
                     .str.title())                   # Title case
```

### Data Validation
```python
def validate_data(df):
    """Data validation function."""
    issues = []
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"Missing values in columns: {missing_cols}")
    
    # Check for duplicates
    if df.duplicated().any():
        issues.append(f"Found {df.duplicated().sum()} duplicate rows")
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col])
                issues.append(f"Column '{col}' should be numeric")
            except:
                pass
    
    return issues

# Usage
validation_issues = validate_data(df)
for issue in validation_issues:
    print(issue)
```

### Complete Cleaning Pipeline
```python
def clean_dataset(df):
    """Complete data cleaning pipeline."""
    # Make a copy
    df_clean = df.copy()
    
    # 1. Remove completely empty rows/columns
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # 2. Handle duplicates
    df_clean = df_clean.drop_duplicates()
    
    # 3. Fix data types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric
            numeric_col = pd.to_numeric(df_clean[col], errors='ignore')
            if not numeric_col.equals(df_clean[col]):
                df_clean[col] = numeric_col
    
    # 4. Handle missing values (strategy depends on data)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Fill numeric with median
    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Fill categorical with mode
    for col in categorical_cols:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean

# Usage
df_cleaned = clean_dataset(df)
```

## Course Notes

# Common Data Problems

## Data type constraints

```python
sales['Revenue'] = sales['Revenue'].str.strip('$')
sales['Revenue'] = sales['Revenue'].astype('int')
```

## Data range constraints

![image.png](attachment:edfe224f-a2c6-4235-a1b8-d4eab354bf14:image.png)

```python
# Convert to datetime object
user_signups['subscription_date'] = pd.to_datetime(user_signups['subscription_date']).dt.date
```

![image.png](attachment:b5033c74-4b16-494b-88c6-6fabaadb4e01:image.png)

## Uniqueness constraints

### How to find duplicate rows?

```python
# Finding duplicate values
duplicates = height_weight.duplicated()
# To see duplicate values
height_weight[duplicates]
```

![image.png](attachment:74a66ded-13c0-4abf-9791-0997015bf93d:image.png)

### How to treat duplicate values?

![image.png](attachment:1f072172-aef3-4834-a6b1-fead8f483662:image.png)

```python
# Example
# Find duplicates
duplicates = ride_sharing.duplicated('ride_id', keep=False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id','duration','user_birth_year']])

# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()

# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}

# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()

# Find duplicated values again
duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
duplicated_rides = ride_unique[duplicates == True]

# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0
```

# Text & Categorical Data Problems

## Membership Constraints

![image.png](attachment:3d3a616d-6db6-4afb-8ead-1d1c94853970:image.png)

![image.png](attachment:eb47c75e-ade8-40c5-9ebb-77ede3c0bf6b:image.png)

## Categorical Variables

![image.png](attachment:55c92252-9da0-4a25-a05f-3f9c7c7fcdcc:image.png)

![image.png](attachment:85ada25d-8ab4-45c5-9ca9-801a3d6a5c93:image.png)

![image.png](attachment:ebc7863d-b00d-4bad-a916-6b2a19a2a6bb:image.png)

![image.png](attachment:a0a5802a-5434-4f9e-a5d3-eabe5dc656f8:image.png)

![image.png](attachment:6c6fb890-c4cf-4737-ba29-bf5ae5c292fb:image.png)

```python
# Example
# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower() 
airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})

# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()

# Verify changes have been effected
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())
```

## Cleaning text data

```python
# Example
# Replace "Dr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Dr.","")

# Replace "Mr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Mr.","")

# Replace "Miss" with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Miss","")

# Replace "Ms." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Ms.","")

# Assert that full_name has no honorifics
assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False
```

# Advanced Data Problems

## Uniformity

```python
# Example: Treating temperature data
temp_fah = temperatures.loc[temperatures['Temperature'] > 40, 'Temperature']
temp_cels = (temp_fah - 32) * (5/9)
temperatures.loc[temperatures['Temperatures'] > 40, 'Temperature'] = temp_cels
```

```python
# Example: Treating date data
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday']) -> # It raises ValueError

birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'], errors = 'coerce') -> # It works
birthdays['Birthday'] = birthdays['Birthday'].dt.strftime('%d-%m-%Y')
```

## Cross field validation

The use of multiple fields in a dataset to sanity check data integrity.

```python
sum_classes = flight[['economy_class', 'business_class', 'first_class']].sum(axis = 1)
passenger_equ = sum_classes == flights['total_passengers']
# Find and filter out rows with inconsistent passenger totals
inconsistent_pass = flights[~passenger_equ]
consistent_pass = flights[passenger_equ]
```

```python
# Example
# Store today's date and find ages
today = dt.date.today()
ages_manual = today.year - banking['birth_date'].dt.year

# Find rows where age column == ages_manual
age_equ = ages_manual == banking['age']

# Store consistent and inconsistent data
consistent_ages = banking[age_equ]
inconsistent_ages = banking[~age_equ]

# Store consistent and inconsistent data
print("Number of inconsistent ages: ", inconsistent_ages.shape[0])
```

## Completeness

```python
import missingno as msno
import matplotlib.pyplot as plt
# Visualize missigness
msno.matrix(airquality)
plt.show()
```

Missingness types:

- Missing completely at random (MCAR) : No systematic relationship between missing data and other values. For example, data entry errors when inputting data
- Missing at random (MAR): Systematic relationship between missing data and other observed values. For example, missing ozone data for high temperatures
- Missing not at random (MNAR): Systematic relationship between missing data and unobserved values. For example, missing temperature values for high temperatures.

### How to deal with missing data?

1. Drop missing data (Simple)
2. Impute with statistical measures (mean, median, mode..) (Simple)
3. Imputing using an algorithmic approach (Complex)
4. Impute with machine learning models (Complex)

```python
# Drop missing values
airquality_dropped = airquality.dropna(subset = ['CO2'])

# Impute with statistical measures
co2_mean = airquality['CO2'].mean()
airquality_imputed = airquality.fillna({'CO2': co2_mean})
```

# Record linkage

## Comparing strings

### Minimum edit distance

Least possible amount of steps needed to transition from one string to another.

```python
from thefuzz import fuzz

# Compare reeding vs reading
fuzz.WRatio('Reeding','Reading') -> 86
```

```python
from thefuzz import process

for state in categories['state']:
		matches = process.extract(state, survey['state'], limit = survey.shape[0])
		for potential_match in matches:
				if potential_match[1] > 80:
						survey.loc[survey['state'] == potential_match[0], 'state'] = state 
```

## Generating pairs

```python
# Import recordlinkage
import recordlinkage

# Create indexing object
indexer = recordlinkage.Index()

# Generate pairs blocked on state
indexer.block('state')
pairs = indexer.index(census_A, census_B)
```