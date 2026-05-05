# Preprocessing for Machine Learning in Python

## Course Overview
Comprehensive course covering data preprocessing techniques for machine learning, including handling missing data, standardization, feature engineering for numerical/categorical/text features, and feature selection methods to prepare datasets for modeling.

## Key Topics Covered

### 1. Introduction to Data Preprocessing
- Removing missing data
- Working with data types
- Training and test sets (with stratified sampling)

### 2. Standardizing Data
- Standardization concept
- Log normalization
- Scaling for feature comparison
- Standardized data and modeling (avoiding data leakage)

### 3. Feature Engineering
- Encoding categorical variables (label & one-hot)
- Engineering numerical features (aggregations, dates)
- Engineering text features (regex, TF-IDF)

### 4. Selecting Features for Modeling
- Removing redundant features
- Selecting features using text vectors
- Dimensionality reduction (PCA)

## Key Concepts

### Train/Test Split with Stratified Sampling
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
```

### Standard Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

### One-Hot Encoding
```python
pd.get_dummies(users['fav_color'])
```

### TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
text_tfidf = tfidf_vec.fit_transform(documents)
```

### PCA for Dimensionality Reduction
```python
from sklearn.decomposition import PCA

pca = PCA()
df_pca = pca.fit_transform(df)
print(pca.explained_variance_ratio_)
```

## Course Notes

# Introduction to Data Preprocessing

## Removing Missing Data

```python
# Dropping all rows which contains NaN value
df.dropna()

# Dropping specific rows from data
df.drop([1, 2, 3])  # It will drop rows 1, 2 and 3

# Dropping specific columns from data
df.drop('A', axis=1)  # It will drop column name A

# Dropping rows from specific column
df.dropna(subset=['B'])  # It will drop rows which has NaN value in column B

# Threshold for NaN values
df.dropna(thresh=2)  # It will drop rows which has NaN values more than or equal to threshold
```

```python
# Example
# Drop the Latitude and Longitude columns from volunteer
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Drop rows with missing category_desc values from volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# Print out the shape of the subset
print(volunteer_subset.shape)
```

## Working with Data Types

```python
# Converting data type from object to float
df['C'] = df['C'].astype('float')
```

```python
# Example
# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer['hits'].astype('int')

# Look at the dtypes of the dataset
print(volunteer.dtypes)
```

## Training and Test Sets

Splitting data will reduce overfitting and lets you evaluate performance on a holdout set.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

For imbalanced classes we can use stratified sampling.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
```

```python
# Example
# Create a DataFrame with all columns except category_desc
X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the y dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# Print the category_desc counts from y_train
print(y_train['category_desc'].value_counts())
```

# Standardizing Data

## Standardization

Standardization means transforming continuous data to appear normally distributed. scikit-learn models assume normally distributed data. Using non-normal training data can introduce bias. Standardization only applies to continuous numerical data.

## Log Normalization

It is useful for features with high variance. It applies a logarithm transformation using the natural log (constant e). Log normalization captures relative changes, the magnitude of change, and keeps everything positive.

```python
df['log_2'] = np.log(df['col2'])
```

```python
# Example
# Print out the variance of the Proline column
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())
```

## Scaling Data for Feature Comparison

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

```python
# Example
from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Subset the DataFrame you want to scale
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to wine_subset
wine_subset_scaled = scaler.fit_transform(wine_subset)
```

## Standardized Data and Modeling

Data Leakage: non-training data is used to train the model. To avoid leakage, fit the scaler on the training set only and transform the test set with that fitted scaler.

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

knn = KNeighborsClassifier()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)
```

# Feature Engineering

Feature engineering is the creation of new features from existing ones. It can improve model performance and reveal relationships between features.

## Encoding Categorical Variables

```python
users['sub_enc'] = users['subscribed'].apply(lambda val: 1 if val == 'y' else 0)
```

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
users['sub_enc_le'] = le.fit_transform(users['subscribed'])
```

```python
# One-Hot Encoding
pd.get_dummies(users['fav_color'])
```

```python
# Example
# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())
```

## Engineering Numerical Features

```python
temps['mean'] = temps.loc[:, 'day1':'day3'].mean(axis=1)
```

```python
purchases['date_converted'] = pd.to_datetime(purchases['date'])
purchases['month'] = purchases['date_converted'].dt.month
```

```python
# Example
# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer['start_date_date'])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer['start_date_converted'].dt.month

# Take a look at the converted and new month columns
print(volunteer[['start_date_converted', 'start_date_month']].head())
```

## Engineering Text Features

Regular expressions are code patterns used to identify text patterns.

```python
import re

my_string = 'temperature:75.6 F'
temp = re.search('\d+\.\d+', my_string)

# \d+ means we want to grab as many digits as possible
# \. matches the literal decimal point
```

TF/IDF (Term Frequency / Inverse Document Frequency) vectorizes words based upon their importance in a document relative to the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
text_tfidf = tfidf_vec.fit_transform(documents)
```

```python
# Example
# Write a pattern to extract numbers and decimals
def return_mileage(length):
    # Search the text for matches
    mile = re.search('\d+\.\d+', length)

    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))

# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking['Length'].apply(return_mileage)
print(hiking[["Length", "Length_num"]].head())
```

# Selecting Features for Modeling

Feature selection is used for modeling. It does not create new features, it picks the most relevant ones to improve model performance.

## Removing Redundant Features

Redundant features include noisy features, correlated features, and duplicated features.

Statistically correlated: features move together directionally. Linear models assume feature independence, so highly correlated features should be reduced.

```python
# Example
# Print out the column correlations of the wine dataset
print(wine.corr())

# Drop that column from the DataFrame
wine = wine.drop('Flavanoids', axis=1)

print(wine.head())
```

## Selecting Features Using Text Vectors

```python
tfidf_vec.vocabulary_
text_tfidf[3].data
text_tfidf[3].indices
```

```python
vocab = {v: k for k, v in tfidf_vec.vocabulary_.items()}

zipped_row = dict(zip(text_tfidf[3].indices, text_tfidf[3].data))
```

```python
def return_weights(vocab, vector, vector_index):
    zipped = dict(zip(vector[vector_index].indices,
                      vector[vector_index].data))

    return {vocab[i]: zipped[i] for i in vector[vector_index].indices}

print(return_weights(vocab, text_tfidf, 3))
```

```python
# Example
# Add in the rest of the arguments
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]: zipped[i] for i in vector[vector_index].indices})

    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, 8, 3))


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)

    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)


# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# Filter the columns in text_tfidf to only those in filtered_words
filtered_text = text_tfidf[:, list(filtered_words)]

# Split the dataset according to the class distribution of category_desc
X_train, X_test, y_train, y_test = train_test_split(
    filtered_text.toarray(), y, stratify=y, random_state=42
)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))
```

## Dimensionality Reduction

Dimensionality reduction is an unsupervised learning method. It combines/decomposes a feature space and is a form of feature extraction.

```python
from sklearn.decomposition import PCA

pca = PCA()
df_pca = pca.fit_transform(df)

print(pca.explained_variance_ratio_)
```

```python
# Example
# Instantiate a PCA object
pca = PCA()

# Define the features and labels from the wine dataset
X = wine.drop('Type', axis=1)
y = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# Apply PCA to the wine dataset X vector
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)

# Fit knn to the training data
knn.fit(pca_X_train, y_train)

# Score knn on the test data and print it out
knn.score(pca_X_test, y_test)
```

## Chapter Example: UFO Sightings End-to-End

```python
# Print the DataFrame info
print(ufo.info())

# Change the type of seconds to float
ufo["seconds"] = ufo['seconds'].astype('float')

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo['date'])

# Count the missing values in the length_of_time, state, and type columns, in that order
print(ufo[['length_of_time', 'state', 'type']].isna().sum())

# Drop rows where length_of_time, state, or type are missing
ufo_no_missing = ufo.dropna(subset=["length_of_time", "state", "type"])

# Print out the shape of the new dataset
print(ufo_no_missing.shape)


def return_minutes(time_string):
    # Search for numbers in time_string
    num = re.search('\d+', time_string)
    if num is not None:
        return int(num.group(0))


# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(return_minutes)

# Take a look at the head of both of the columns
print(ufo[['minutes', 'length_of_time']].head())

# Check the variance of the seconds and minutes columns
print(ufo[['seconds', 'minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())

# Use pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda val: 1 if val == "us" else 0)

# Print the number of unique type values
print(len(ufo["type"].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

# Look at the first 5 rows of the date column
print(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].dt.year

# Take a look at the head of all three columns
print(ufo[['date', 'month', 'year']].head())

# Take a look at the head of the desc field
print(ufo['desc'].head())

# Instantiate the tfidf vectorizer object
vec = TfidfVectorizer()

# Fit and transform desc using vec
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns and rows
print(desc_tfidf.shape)

# Make a list of features to drop
to_drop = ['city', 'country', 'date', 'desc', 'lat', 'length_of_time',
           'long', 'minutes', 'recorded', 'seconds', 'state']

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# Fit knn to the training sets
knn.fit(X_train, y_train)

# Print the score of knn on the test sets
print(knn.score(X_test, y_test))

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y
X_train, X_test, y_train, y_test = train_test_split(
    filtered_text.toarray(), y, stratify=y, random_state=42
)

# Fit nb to the training sets
nb.fit(X_train, y_train)

# Print the score of nb on the test sets
print(nb.score(X_test, y_test))
```

## Key Techniques Covered

### Missing Data Handling
1. **Row/column dropping**: `dropna()`, `drop()`, `dropna(subset=...)`, `dropna(thresh=...)`
2. **Type conversion**: `astype()` for object → numeric/datetime conversions

### Standardization
1. **Log normalization**: For high-variance features
2. **StandardScaler**: Zero mean, unit variance scaling
3. **Leakage-safe pipeline**: `fit_transform` on train, `transform` on test

### Feature Engineering
1. **Lambda encoding**: Custom binary encoding for categorical variables
2. **LabelEncoder**: Sklearn ordinal encoding
3. **One-hot encoding**: `pd.get_dummies()` for nominal categories
4. **Numerical aggregation**: Row-wise mean/sum across columns
5. **Datetime extraction**: Month, year, day from date columns
6. **Regex extraction**: Pulling numbers/patterns out of text
7. **TF-IDF**: Vectorizing text by term importance

### Feature Selection
1. **Correlation-based removal**: Drop redundant correlated features
2. **TF-IDF top-N filtering**: Keep most informative words per document
3. **PCA**: Variance-preserving linear dimensionality reduction

## Skills Demonstrated

### Data Cleaning
- Handling missing values strategically
- Converting between data types
- Filtering and subsetting datasets

### Modeling Preparation
- Train/test splitting with stratification
- Avoiding data leakage in scaling pipelines
- Applying transformations consistently across splits

### Advanced Engineering
- Building text features with regex and TF-IDF
- Combining engineered features with existing ones
- Reducing dimensionality with PCA

## Important Concepts

### Data Leakage
Information from outside the training dataset leaking into model training. Always fit transformers (scalers, encoders) on the training set only.

### Bias from Non-Normal Data
Many sklearn models assume normally distributed inputs. Standardization or log-normalization mitigates this assumption violation.

### Sparse Text Representations
TF-IDF produces sparse matrices where most entries are zero. Indices and data attributes give efficient access to non-zero terms.

## Key Takeaways

- **Drop strategically**: Remove rows/columns only when missingness can't be informative
- **Always split before scaling**: Prevent data leakage from test set into training pipeline
- **Stratify on imbalanced targets**: Preserve class proportions in train/test splits
- **Log-normalize high-variance features**: Bring them onto a comparable scale
- **One-hot vs label encoding**: Use one-hot for nominal categories, label encoding for ordinal
- **Datetime is a feature gold mine**: Extract month, year, day-of-week, etc.
- **TF-IDF beats raw counts**: It downweights ubiquitous, low-information words
- **Drop correlated features**: Linear models assume independence
- **PCA reduces dimensions**: Useful when many features carry overlapping information
