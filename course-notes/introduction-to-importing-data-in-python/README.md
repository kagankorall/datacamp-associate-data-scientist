# Introduction to Importing Data in Python

## Course Overview
This course covers various methods for importing different types of data into Python, including flat files, databases, web data, and APIs.

## Key Topics Covered

### 1. Flat File Imports
- Reading CSV files
- Excel files
- Text files
- JSON files
- Handling different delimiters and encodings

### 2. Database Connections
- SQLite databases
- MySQL and PostgreSQL
- SQL queries in Python
- Database connections with SQLAlchemy

### 3. Web Data
- Web scraping basics
- API data retrieval
- HTML parsing
- HTTP requests

### 4. Other Data Sources
- Pickle files
- HDF5 files
- Statistical software files (SAS, STATA)
- ZIP archives

## Key Concepts

### CSV and Text Files
```python
import pandas as pd
import numpy as np

# Basic CSV reading
df = pd.read_csv('data.csv')

# Custom parameters
df = pd.read_csv('data.csv', 
                 sep=';',           # Custom separator
                 header=0,          # Header row
                 names=['col1', 'col2'],  # Custom column names
                 skiprows=1,        # Skip rows
                 na_values=['NULL', 'N/A'],  # Custom NA values
                 encoding='utf-8')  # Encoding

# Reading large files in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    print(chunk.shape)
```

### Excel Files
```python
# Read Excel file
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Multiple sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)

# Specific range
df = pd.read_excel('data.xlsx', usecols='A:C', nrows=100)

# Writing to Excel
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

### JSON Files
```python
import json

# Read JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Using pandas for JSON
df = pd.read_json('data.json')

# Normalized JSON (nested structures)
df = pd.json_normalize(data)

# Write JSON
df.to_json('output.json', orient='records', indent=2)
```

### Database Connections
```python
import sqlite3
from sqlalchemy import create_engine

# SQLite connection
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
conn.close()

# Using SQLAlchemy
engine = create_engine('sqlite:///database.db')
df = pd.read_sql_query("SELECT * FROM table_name", engine)

# PostgreSQL example
engine = create_engine('postgresql://user:password@localhost:5432/database')
df = pd.read_sql_query("SELECT * FROM table_name", engine)
```

### Web Scraping and APIs
```python
import requests
from bs4 import BeautifulSoup

# Simple HTTP request
response = requests.get('https://api.example.com/data')
data = response.json()

# Web scraping
url = 'https://example.com'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract data from HTML
tables = soup.find_all('table')
df = pd.read_html(str(tables[0]))[0]

# API with authentication
headers = {'Authorization': 'Bearer token'}
response = requests.get('https://api.example.com/data', headers=headers)
```

### Pickle Files
```python
import pickle

# Save to pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(df, f)

# Load from pickle
with open('data.pkl', 'rb') as f:
    df = pickle.load(f)

# Using pandas
df.to_pickle('data.pkl')
df = pd.read_pickle('data.pkl')
```

### File Path Handling
```python
import os
from pathlib import Path

# Current working directory
current_dir = os.getcwd()
print(current_dir)

# List files in directory
files = os.listdir('data/')
csv_files = [f for f in files if f.endswith('.csv')]

# Using pathlib (modern approach)
data_path = Path('data')
csv_files = list(data_path.glob('*.csv'))

# Read multiple files
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df['source_file'] = file.name
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
```

### Error Handling
```python
def safe_read_csv(filepath):
    """Safely read CSV with error handling."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Empty file: {filepath}")
        return None
    except pd.errors.ParserError as e:
        print(f"Parsing error in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
df = safe_read_csv('data.csv')
if df is not None:
    print(df.head())
```

## Course Notes

## Introduction and Flat Files

### Reading a text file

```python
filename = 'huck_finn.txt'
file = open(filename, mode='r') # r is to read
text = file.read()
file.close()

file = open(filename, mode='w') # w is to read
```

```python
# Example
# Open a file: file
with open('moby_dick.txt', 'r') as file:
    print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)
```

```python
# Example
# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
```

### Flat Files

.csv files are flat files. Text files containing records. That is, table data.

- Record: row of fields or attributes
- Column: Feature or attribute

### Importing flat files using NumPy

![image.png](attachment:3c62c458-fad1-4c02-9d4e-d63d8abc6d90:image.png)

![image.png](attachment:6d236b7a-6a26-412f-85d5-150e38d16583:image.png)

```python
# Example
# Import package
import numpy as np

# Assign filename to variable: file
file = 'digits.csv'

# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')

# Print datatype of digits
print(type(digits))

# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()

```

### Importing flat files using pandas

```python
# Assign the filename: file
file = 'digits.csv'

# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header=None)

# Build a numpy array from the DataFrame: data_array
data_array = data.values

# Print the datatype of data_array to the shell
print(type(data_array))
```

```python
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Assign filename: file
file = 'titanic_corrupt.txt'

# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')

# Print the head of the DataFrame
print(data.head())

# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()

```

## Introduction to other file types

### Pickle Files

```python
import pickle
with open('pickle_fruit.pkl', 'rb') as file:
		data = pickle.load(file)
```

### Excel Spreadsheets

```python
import pandas as pd
file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)

df1 = data.parse('1960-1966') # sheet name as a string
df2 = data.parse(0) # sheet index, as float
```

```python
# Parse the first sheet and rename the columns: df1
df1 = xls.parse(0, skiprows=[0], names=['Country','AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xls.parse(1, usecols=[0], skiprows=[0], names=['Country'])

# Print the head of the DataFrame df2
print(df2.head())
```

### SAS and Stata files

SAS: Statistical Analysis System

SAS is for business analytics and biostatistics. On the other hand, Stata files are for academic social sciences research

```python
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sas7bdat') as file:
		df_sas = file.to_data_frame()
```

```python
import pandas as pd
data = pd.read_stata('urbanpop.dta')
```

### HDF5 files

Hierarchical Data Format version 5

Standard for storing large quantities of numerical data

![image.png](attachment:dced0deb-d992-4d27-9cae-9c81b27cb303:image.png)

## Introduction to Relational Databases

### Creating a database engine

```python
from sqlalchemy import create_engine
engine = create_engine('sqlite://Northwind.sqlite')

table_names = engine.table_names()
```

### Querying relational databases in Python

```python
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite://Northwind.sqlite')
con = engine.connect()
rs = con.execute("SELECT * FROM Orders")
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()

# OR
df = pd.read_sql_query("SELECT * FROM Orders", engine)
```

```python
# Example
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title FROM Employee')
    df = pd.DataFrame(rs.fetchmany(size=3))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print(len(df))

# Print the head of the DataFrame df
print(df.head())
```

```python
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("select Title,Name from album inner join artist on album.ArtistID = artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print head of DataFrame df
print(df.head())
```