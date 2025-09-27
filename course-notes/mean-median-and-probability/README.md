# Mean, Median and Probability

## Course Overview
This course covers fundamental statistical concepts including measures of central tendency and basic probability theory.

## Key Topics Covered

### 1. Measures of Central Tendency
- Mean (Average)
- Median 
- Mode
- When to use each measure

### 2. Probability Fundamentals
- Basic probability rules
- Conditional probability
- Independence

### 3. Practical Applications
- Real-world examples
- Python implementations

## Key Concepts

### Mean
```python
import numpy as np
data = [1, 2, 3, 4, 5]
mean = np.mean(data)
print(f"Mean: {mean}")
```

### Median
```python
median = np.median(data)
print(f"Median: {median}")
```

## Course Notes

If the distribution of the data is symmetric, use the **mean** as the measure of central tendency. However, if the distribution is left or right skewed, use the **median** instead, as the mean is sensitive to outlier values and can be misleading in skewed distributions.

In symmetric data, the mean and median values are close to each other, making either measure appropriate. However, in asymmetric (skewed) data, there is a significant difference between the mean and median values.

The mean is affected by the direction of the skew in the data distribution:

- In **right-skewed** (positively skewed) distributions, the mean is pulled toward the higher values and is greater than the median
- In **left-skewed** (negatively skewed) distributions, the mean is pulled toward the lower values and is less than the median
- The median remains more stable and better represents the typical value in skewed distributions

This is why the median is often preferred for skewed data, as it provides a more robust measure of central tendency that is not influenced by extreme values or outliers.

### Probability

Two events are independent if the probability of the second event is not affected by the outcome of the first event. On the other hand, if outcome of the first changes outcome of the second it means dependent event.

Sampling with replacement = each pick is independent (Rolling dice or flipping coin)

Sampling without replacement = each pick is dependent

`df.sample()` → Picks a random data from the dataframe

`np.random.seed(n)` → Generates n random sample