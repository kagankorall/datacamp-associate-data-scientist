# Working with Categorical Data in Python

## Course Overview
This course covers techniques for handling, analyzing, and visualizing categorical data in Python using pandas and other libraries.

## Key Topics Covered

### 1. Categorical Data Types
- Understanding categorical data
- Pandas categorical dtype
- Ordered vs unordered categories
- Category creation and manipulation

### 2. Data Encoding Techniques
- Label encoding
- One-hot encoding
- Target encoding
- Binary encoding

### 3. Categorical Data Analysis
- Frequency analysis
- Cross-tabulation
- Chi-square tests
- Association measures

### 4. Visualization
- Bar plots and count plots
- Stacked bar charts
- Mosaic plots
- Heatmaps for categorical relationships

## Key Concepts

### Creating Categorical Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to categorical
df['category'] = df['category'].astype('category')

# Create categorical with specific categories
categories = ['Low', 'Medium', 'High']
df['level'] = pd.Categorical(df['level'], categories=categories, ordered=True)

# Check categorical info
print(df['category'].cat.categories)
print(df['category'].cat.codes)
```

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

# Label encoding
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Manual mapping
category_map = {'A': 0, 'B': 1, 'C': 2}
df['category_manual'] = df['category'].map(category_map)
```

### One-Hot Encoding
```python
# Using pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Using sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
encoded_array = ohe.fit_transform(df[['category']])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(['category']))
```

### Frequency Analysis
```python
# Value counts
category_counts = df['category'].value_counts()
print(category_counts)

# Proportions
category_props = df['category'].value_counts(normalize=True)
print(category_props)

# Cross-tabulation
cross_tab = pd.crosstab(df['category1'], df['category2'])
print(cross_tab)

# Normalized cross-tab
cross_tab_norm = pd.crosstab(df['category1'], df['category2'], normalize='index')
```

### Categorical Data Visualization
```python
# Count plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.countplot(data=df, x='category')
plt.title('Category Counts')
plt.xticks(rotation=45)

# Stacked bar chart
plt.subplot(2, 2, 2)
cross_tab.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Stacked Bar Chart')

# Heatmap of cross-tabulation
plt.subplot(2, 2, 3)
sns.heatmap(cross_tab, annot=True, cmap='Blues')
plt.title('Category Cross-tabulation Heatmap')

# Proportion plot
plt.subplot(2, 2, 4)
cross_tab_norm.plot(kind='bar', ax=plt.gca())
plt.title('Proportional Distribution')

plt.tight_layout()
plt.show()
```

### Statistical Tests
```python
from scipy.stats import chi2_contingency

# Chi-square test for independence
chi2, p_value, dof, expected = chi2_contingency(cross_tab)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")

# Cramér's V (effect size)
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

v = cramers_v(cross_tab)
print(f"Cramér's V: {v}")
```

### Advanced Encoding Techniques
```python
# Target encoding (for supervised learning)
def target_encode(df, column, target):
    target_mean = df.groupby(column)[target].mean()
    df[f'{column}_target_encoded'] = df[column].map(target_mean)
    return df

# Binary encoding
def binary_encode(df, column):
    from category_encoders import BinaryEncoder
    be = BinaryEncoder(cols=[column])
    df_encoded = be.fit_transform(df)
    return df_encoded

# Frequency encoding
def frequency_encode(df, column):
    freq_map = df[column].value_counts().to_dict()
    df[f'{column}_freq_encoded'] = df[column].map(freq_map)
    return df
```

## Course Notes

# Introduction to Categorical Data

## Categorical vs. Numerical Data

Categorical means finite number or groups. These categories are usually fixed or known (eye color, hair color, etc.). Known as qualitative data. On the other hand, numerical data is known as quantitative data. Expressed using a numerical value. Is usually a measurement (height, weight, IQ, etc.)

## Ordinal vs. Nominal variables

Ordinal variables are categorical variables that have a natural order such as strongly disagree, disagree, neutral, agree and strongly agree.

Nominal variables are categorical variables that cannot be placed into a natural order. For example: blue green red yellow purple

```python
# Gives number of data, number of unique labels, top frequent label
adult["Marital Status"].describe()
```

## Categorical data in pandas

```python
adult = pd.read_csv("adult.csv")
adult.dtypes

adult["Marital Status"] = adult["Marital Status"].astype("category")
```

```python
# Creating a categorical series
my_data = ["A", "A", "C", "B", "C", "A"]

my_series1 = pd.Series(my_data, dtype = "category")
my_series2 = pd.Categorical(my_data, categories=["C","B","A"], ordered = True)
```

Categorical data is memory saver.

![image.png](attachment:b2ad6c9c-9ed9-425e-ab89-29626a40dd18:image.png)

![image.png](attachment:67af7792-60c9-44b1-86d1-0c25b91ad338:image.png)

## Grouping data by category

```python
groupby_object = adult.groupby(by=["Above/Below 50k"])

groupby_object.mean() = adult.groupby(by=["Above/Below 50k"]).mean()
```

![image.png](attachment:5c5b334e-fb30-4eaf-9d63-b4d90c6d934c:image.png)

Option 1 is preferred especially in large datasets.

![image.png](attachment:20a2e18b-09bb-42ab-a970-fd867c8debc6:image.png)

```python
# Example
# Group the adult dataset by "Sex" and "Above/Below 50k"
gb = adult.groupby(by=["Sex","Above/Below 50k"])

# Print out how many rows are in each created group
print(gb.size())

# Print out the mean of each group for all columns
print(gb.mean())

# Create a list of user-selected variables
user_list = ["Education","Above/Below 50k"]

# Create a GroupBy object using this list
gb = adult.groupby(by=user_list)

# Find the mean for the variable "Hours/Week" for each group - Be efficient!
print(gb["Hours/Week"].mean())
```

# Categorical pandas Series

## Setting category variables

```python
Series.cat.method_name
```

Common parameters:

- new_categories: a list of categories
- inplace: Boolean - whether or not the update should overwrite the Series
- ordered: Boolean - whether or not the categorical is treated as an ordered categorical

```python
# Set categories
dogs["coat"] = dogs["coat"].cat.set_categories(
	new_categories = ["short","medium","long],
	ordered=True)
```

```python
# Adding categories
dogs["likes_people"] = dogs["likes_people"].astype("category")
dogs["likes_people"] = dogs["likes_people"].cat.add_categories(
	new_categories = ["did not check", "could not tell"])
```

```python
# Removing categories
dogs["coat"] = dogs["coat"].astype("category")
dogs["coat"] = dogs["coat"].cat.remove_categories(removals=["wirehaired"])
```

```python
#Example
# Check frequency counts while also printing the NaN count
print(dogs["keep_in"].value_counts(dropna=False))

# Switch to a categorical variable
dogs["keep_in"] = dogs["keep_in"].astype("category")

# Add new categories
new_categories = ["Unknown History", "Open Yard (Countryside)"]
dogs["keep_in"] = dogs["keep_in"].cat.add_categories(new_categories)

# Check frequency counts one more time
print(dogs["keep_in"].value_counts(dropna=False))
```

## Updating categories

```python
# Renaming categories
Series.cat.rename_categories(new_categories=dict)
```

```python
# Example
my_changes = {"Unknown Mix": "Unknown"}

dogs["breed"] = dogs["breed"].cat.rename_categories(my_changes)
```

```python
# By using a function
dogs["sex"] = dogs["sex"].cat.rename_categories(lambda c: c.title())
```

```python
# Collapsing categories
update_colors = {
			"black and brown": "black",
			"black and tan": "black",
			"black and white": "black"}

dogs["main_color"] = dogs["color"].replace(update_colors)
```

```python
# Example
# Create the my_changes dictionary
my_changes = {
    "Maybe?": "Maybe"
}

# Rename the categories listed in the my_changes dictionary
dogs["likes_children"] = dogs["likes_children"].cat.rename_categories(my_changes)

# Use a lambda function to convert all categories to uppercase using upper()
dogs["likes_children"] =  dogs["likes_children"].cat.rename_categories(lambda c: c.upper())

# Print the list of categories
print(dogs["likes_children"].cat.categories)
```

```python
#Example
# Create the update_coats dictionary
update_coats = {
    "wirehaired": "medium",
    "medium-long": "medium"
}

# Create a new column, coat_collapsed
dogs["coat_collapsed"] = dogs["coat"].replace(update_coats)

# Convert the column to categorical
dogs["coat_collapsed"] = dogs["coat_collapsed"].astype("category")

# Print the frequency table
print(dogs["coat_collapsed"].value_counts())
```

## Reordering categories

Why would you reorder?

1. Creating a ordinal variable
2. To set the order that variables are displayed in anaylsis
3. Memory savings

```python
dogs["coat"] = dogs["coat"].cat.reorder_categories(
	new_categories = ["short", "medium", "wirehaired", "long"],
	ordered = True,
	inplace = True
)
```

## Cleaning and accessing data

Possible issues with categorical data

1. Inconsistent values
2. Misspelled values
3. Wrong dtype

```python
# Fixing issues
# Removing whitespace
dogs["get_along_cats"] = dogs["get_along_cats"].str.strip()
# Fixing capitalization
dogs["get_along_cats"] = dogs["get_along_cats"].str.title()
# Misspelled words
replace_map = {"Noo": "No"}
dogs["get_along_cats"].replace(replace_map, inplace = True)
```

![image.png](attachment:3848ad91-f6c1-4acd-9d15-a34be726f072:image.png)

# Visualizing Categorical Data

## Categorical plots in Seaborn

sns.catplot Parameters:

- x: name of variable
- y: name of variable
- data
- kind: type of plot like ‘box’, ‘violin’, ’point’ etc.

```python
sns.catplot(
	x='Pool',
	y='Score',
	data=reviews,
	kind='box'
)
plt.show()

sns.set(font_scale=1.4)
sns.set_style('whitegrid')
```

```python
# Bar Plots
reviews['Traveler type'].value_counts().plot.bar()

sns.catplot(x='Traveler type', y='Score', data=reviews, kind='bar', hue='Tennis court')
plt.show()
```

```python
#Example
# Set style
sns.set(font_scale=.9)
sns.set_style("whitegrid")

# Print the frequency counts for "User continent"
print(reviews["User continent"].value_counts())

# Convert "User continent" to a categorical variable
reviews["User continent"] = reviews["User continent"].astype("category")

# Reorder "User continent" using continent_categories and rerun the graphic
continent_categories = list(reviews["User continent"].value_counts().index)
reviews["User continent"] = reviews["User continent"].cat.reorder_categories(new_categories=continent_categories)
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar")
plt.show()
```

```python
# Point Plot
sns.catplot(x='Spa', y='Score', data=reviews, kind='point', hue='Tennis court',
 dodge=True, -> # lines do not overlap
join = False) -> # lines do not join
```

![image.png](attachment:f726b861-f372-41ab-af99-537ac333529b:image.png)

![image.png](attachment:0d5df48c-e9e8-4227-998e-a115ffc737a8:image.png)

```python
# Example
# Create a catplot for each "Period of stay" broken down by "Review weekday"
ax = sns.catplot(
  # Make sure Review weekday is along the x-axis
  x = 'Review weekday',
  # Specify Period of stay as the column to create individual graphics for
  col = 'Period of stay',
  # Specify that a count plot should be created
  kind = 'count',
  # Wrap the plots after every 2nd graphic.
  col_wrap = 2,
  data=reviews
)
plt.show()
```

```python
# Example
# Adjust the color
ax = sns.catplot(
  x="Free internet", y="Score",
  hue="Traveler type", kind="bar",
  data=reviews,
  palette=sns.color_palette("Set2")
)

# Add a title
ax.fig.suptitle("Hotel Score by Traveler Type and Free Internet Access")
# Update the axis labels
ax.set_axis_labels("Free Internet", "Average Review Rating")

# Adjust the starting height of the graphic
plt.subplots_adjust(top=0.93)
plt.show()
```

# Pitfalls and Encoding

## Categorical pitfalls

Using the .str accessor object to manipulate data converts the Series to an object.

The .apply() method outputs a new Series as an object.

The common methods of adding, removing, replacing, or setting categories do not all handle missing categories the same way.

NumPy functions generally do not work with categorical Series.

```python
# Example
# Print the frequency table of body_type and include NaN values
print(used_cars["body_type"].value_counts(dropna=False))

# Update NaN values
used_cars.loc[used_cars["body_type"].isna(), "body_type"] = "other"

# Convert body_type to title case
used_cars["body_type"] = used_cars["body_type"].str.title()

# Check the dtype
print(used_cars['body_type'].dtype)
```

## Label Encoding

It codes each category as an integer from 0 through n-1, where n is the number of categories. A -1 code is reserved for any missing values. It can save memory and often used in surveys.

```python
used_cars['manufacturer_name'] = used_cars['manufacturer_name'].astype('category')
used_cars['manufacturer_code'] = used_cars['manufacturer_name'].cat.codes
```

```python
codes = used_cars['manufacturer_name'].cat.codes
categories = used_cars['manufacturer_name']

name_map = dict(zip(codes, categories))

# To revert previous values
used_cars['manufacturer_code'].map(name_map)

# Example
# Convert to categorical and print the frequency table
used_cars["color"] = used_cars["color"].astype("category")
print(used_cars["color"].value_counts())

# Create a label encoding
used_cars["color_code"] = used_cars["color"].cat.codes

# Create codes and categories objects
codes = used_cars["color"].cat.codes
categories = used_cars["color"]
color_map = dict(zip(codes, categories))

# Print the map
print(used_cars['color_code'].map(color_map))
```

```python
# Boolean coding
# Find all body types that have "van" in them:
used_cars['body_type'].str.contains("van", regex=False)

used_cars['van_code'] = np.where(
	used_cars['body_type'].str.contains('van', regex=False),1,0)
```

## One Hot Encoding

pd.get_dummies()

- data: a pandas dataframe
- columns: a list like object of column names
- prefix: a string to add to the beginning of each category

```python
# Example
used_cars_onehot = pd.get_dummies(used_cars[['odometer_value','color']])
used_cars_onehot.head()

used_cars_onehot = pd.get_dummies(used_cars, columns=['color'], prefix="")
```