# Working with Dates and Times in Python

## Course Overview
This course covers comprehensive techniques for handling date and time data in Python, including parsing, formatting, time zone handling, and time series analysis.

## Key Topics Covered

### 1. DateTime Fundamentals
- Python datetime module
- Pandas datetime functionality
- Date and time parsing
- Formatting dates and times

### 2. Time Zone Handling
- UTC and local times
- Time zone conversions
- Daylight saving time considerations

### 3. Date Arithmetic
- Adding and subtracting time periods
- Date ranges and frequencies
- Business day calculations

### 4. Time Series Analysis
- Time-based indexing
- Resampling and aggregation
- Rolling windows
- Seasonal decomposition

## Key Concepts

### Basic DateTime Operations
```python
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt

# Create datetime objects
current_time = datetime.now()
specific_date = datetime(2024, 12, 25, 15, 30, 0)

# Parse string to datetime
date_string = "2024-01-15 14:30:00"
parsed_date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

# Using pandas
df['date_column'] = pd.to_datetime(df['date_column'])
df['date_column'] = pd.to_datetime(df['date_column'], format='%Y-%m-%d')

# Handle parsing errors
df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
```

### Date Formatting and Extraction
```python
# Format datetime
formatted_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
print(f"Formatted: {formatted_date}")

# Extract components
df['year'] = df['date_column'].dt.year
df['month'] = df['date_column'].dt.month
df['day'] = df['date_column'].dt.day
df['weekday'] = df['date_column'].dt.dayofweek
df['weekday_name'] = df['date_column'].dt.day_name()
df['quarter'] = df['date_column'].dt.quarter

# Common format strings
formats = {
    'ISO format': '%Y-%m-%d',
    'US format': '%m/%d/%Y',
    'European': '%d/%m/%Y',
    'Full datetime': '%Y-%m-%d %H:%M:%S',
    'Month name': '%B %d, %Y'
}
```

### Time Zone Handling
```python
import pytz

# Create timezone-aware datetime
utc_time = datetime.now(pytz.UTC)
eastern = pytz.timezone('US/Eastern')
eastern_time = datetime.now(eastern)

# Convert between timezones
utc_to_eastern = utc_time.astimezone(eastern)

# Using pandas
df['date_utc'] = pd.to_datetime(df['date_column'], utc=True)
df['date_eastern'] = df['date_utc'].dt.tz_convert('US/Eastern')

# Localize naive datetime
df['date_localized'] = df['date_column'].dt.tz_localize('UTC')
```

### Date Arithmetic
```python
# Add/subtract time periods
future_date = current_time + timedelta(days=30)
past_date = current_time - timedelta(weeks=2, days=3)

# Using pandas periods
df['30_days_later'] = df['date_column'] + pd.Timedelta(days=30)
df['1_week_ago'] = df['date_column'] - pd.Timedelta(weeks=1)

# Date ranges
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
business_days = pd.bdate_range(start='2024-01-01', end='2024-12-31')
monthly_range = pd.date_range(start='2024-01-01', periods=12, freq='M')

# Calculate differences
df['days_since'] = (datetime.now() - df['date_column']).dt.days
df['time_diff'] = df['end_date'] - df['start_date']
```

### Time Series Indexing
```python
# Set datetime index
df.set_index('date_column', inplace=True)
df = df.sort_index()

# Select by date
recent_data = df['2024-01-01':'2024-12-31']
january_data = df['2024-01']
specific_day = df['2024-01-15']

# Boolean indexing with dates
last_month = df[df.index > (datetime.now() - timedelta(days=30))]
weekends = df[df.index.dayofweek >= 5]
```

### Resampling and Aggregation
```python
# Resample time series data
daily_avg = df.resample('D').mean()
weekly_sum = df.resample('W').sum()
monthly_max = df.resample('M').max()
quarterly_data = df.resample('Q').agg({
    'sales': 'sum',
    'customers': 'mean',
    'orders': 'count'
})

# Upsampling and downsampling
hourly_data = df.resample('H').interpolate()  # Upsample and interpolate
daily_data = df.resample('D').first()         # Downsample

# Custom resampling
def custom_agg(x):
    return {
        'total': x.sum(),
        'average': x.mean(),
        'count': len(x)
    }

custom_resample = df['value'].resample('M').apply(custom_agg)
```

### Rolling Windows
```python
# Rolling statistics
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_30'] = df['value'].rolling(window=30).std()
df['rolling_sum_14'] = df['value'].rolling(window=14).sum()

# Expanding windows
df['expanding_mean'] = df['value'].expanding().mean()
df['expanding_max'] = df['value'].expanding().max()

# Rolling with custom functions
def rolling_range(x):
    return x.max() - x.min()

df['rolling_range'] = df['value'].rolling(window=10).apply(rolling_range)

# Time-based rolling windows
df['rolling_7d'] = df['value'].rolling('7D').mean()
df['rolling_1M'] = df['value'].rolling('30D').sum()
```

### Seasonal Analysis
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Seasonal decomposition
decomposition = seasonal_decompose(df['value'], model='additive', period=365)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Detect seasonality patterns
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['hour'] = df.index.hour

monthly_pattern = df.groupby('month')['value'].mean()
weekly_pattern = df.groupby('day_of_week')['value'].mean()
hourly_pattern = df.groupby('hour')['value'].mean()
```

### Working with Periods
```python
# Period objects
period = pd.Period('2024-01', freq='M')
next_month = period + 1
quarter = period.asfreq('Q')

# Period ranges
period_range = pd.period_range(start='2024-01', end='2024-12', freq='M')

# Convert between period and timestamp
df['period'] = df.index.to_period('M')
df['timestamp'] = df['period'].dt.to_timestamp()
```

### Handling Missing Dates
```python
# Find missing dates in sequence
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
missing_dates = full_range.difference(df.index)

# Fill missing dates
df_complete = df.reindex(full_range)
df_complete = df_complete.interpolate()  # Interpolate missing values

# Forward fill for missing dates
df_complete = df_complete.fillna(method='ffill')
```

### Time Series Visualization
```python
# Basic time series plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'])
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Multiple time series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
df['value1'].plot(ax=axes[0], title='Series 1')
df['value2'].plot(ax=axes[1], title='Series 2')
plt.tight_layout()
plt.show()

# Seasonal plots
monthly_avg = df.groupby(df.index.month)['value'].mean()
monthly_avg.plot(kind='bar', title='Average by Month')
plt.show()
```

## Course Notes

# Dates and Calenders

## Dates in Python

```python
from datetime import date
two_hurricanes_dates = [date(2016,10,7), date(2017,6,21)]
print(two_hurricane_dates[0].year)
```

## Math with dates

```python
from datetime import date
d1 = date(2017,11,5)
d2 = date(2017,12,4)
l = [d1, d2]
delta = d2 - d1
print(delta.days) -> 29
```

```python
from datetime import timedelta
# Create a 29 day timedelta
td = timedelta(days=29)
print(d1+td) -> 2017-12-04
```

## Turning dates into strings

```python
# Assign the earliest date to first_date
first_date = min(florida_hurricane_dates)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime('%m/%d/%Y')

print("ISO: " + iso)
print("US: " + us)
```

# Combining Dates and Times

```python
# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Use the replace method to change the year
dt_old = dt.replace(year=1917)

# Print the results
print(dt_old.isoformat())
```

```python
# Create dictionary to hold results
trip_counts = {'AM': 0, 'PM': 0}
  
# Loop over all trips
for trip in onebike_datetimes:
  # Check to see if the trip starts before noon
  if trip['start'].hour < 12:
    # Increment the counter for before noon
    trip_counts['AM'] += 1
  else:
    # Increment the counter for after noon
    trip_counts['PM'] += 1
  
print(trip_counts)
```

## Printing and Parsing Dates

```python
# Import the datetime class
from datetime import datetime

# Starting string, in YYYY-MM-DD HH:MM:SS format
s = '2017-02-03 00:00:01'

# Write a format string to parse s
fmt = '%Y-%m-%d %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)
```