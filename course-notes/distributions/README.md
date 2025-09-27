# Distributions

## Course Overview
This course covers probability distributions, their properties, and applications in data science.

## Key Topics Covered

### 1. Probability Distributions
- Discrete distributions
- Continuous distributions
- Distribution properties

### 2. Common Distributions
- Normal distribution
- Binomial distribution
- Poisson distribution
- Uniform distribution

### 3. Distribution Analysis
- Probability density functions
- Cumulative distribution functions
- Parameters and moments

## Key Concepts

### Normal Distribution
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate normal distribution
data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30, density=True, alpha=0.7)
plt.title('Normal Distribution')
plt.show()
```

### Binomial Distribution
```python
# Binomial distribution example
n, p = 10, 0.5
binomial_data = np.random.binomial(n, p, 1000)
plt.hist(binomial_data, bins=range(12), density=True, alpha=0.7)
plt.title('Binomial Distribution')
plt.show()
```

## Course Notes

## **Distributions**

### Discrete Distributions

Expected value = mean of a probability distribution

Countable models

### Normal Distribution

![image.png](attachment:eb50e6f1-8fa1-402d-9d7a-73a7d13b4b70:1aa1cda2-e0c1-4fc6-a0b0-428b97d75ffc.png)

- Symmetrical
- Area beneath the curve = 1
- Curve never hits 0

68% falls within 1 standard deviation

95% falls within 2 standard deviation

99.7% falls within 3 standard deviation

```python
from scipy.stats import norm
# Example: What percent of women are shorter than 154 cm?
norm.cdf(154,161,7)
# Mean = 161 & St. Dev. = 7

# Taller than 154
1 - norm.cdf(154,161,7)

# What height are 90% of women shorter than?
norm.ppf(0.9,161,7)
1 - norm.ppf(0.9,161,7)

# Generate 10 random heights
norm.rvs(161, 7, size = 10)
```

### Central Limit Theorem

```python
die = pd.Series([1,2,3,4,5,6])
# Roll 5 Times
samp_5 = die.sample(5, replace = True)
print(samp_5)
```

When we roll a dice 100 times and calculate the mean of these rolls, the distribution of this mean will resemble a normal distribution. As we increase the number of rolls to 1,000 or 10,000, the distribution of the mean becomes even more closely aligned with a normal distribution. This phenomenon is known as the **Central Limit Theorem**.

In the literature, it is established that the sampling distribution of a statistic approaches a normal distribution as the number of trials increases. The Central Limit Theorem guarantees that regardless of the shape of the population distribution, the sampling distribution of the sample mean will approximate a normal distribution when the sample size is sufficiently large.

This theorem is fundamental in data science because many statistical tests and confidence interval calculations rely on the assumption of normality. 

Note: Samples should be random and independent

### Poisson Distribution

![image.png](attachment:5929960d-4d0b-43a0-8e70-406169ebf666:b7171097-0e9f-41b1-9506-4fc970ea5375.png)

Poisson processes: Events appear to happen at a certain rate, but completely at random. Example: Number of animals adopted from an animal shelter per week. Number of people arriving at a restaurant per hour.

The time unit is irrelevant, as long as you use the same unit when talking about the same situation.

Probability of some number of events occurring over a fixed period of time. It described with Lambda() which means “average number of events per time interval”. Lambda changes the shape of the poisson distribution.

```python
# Example
# If the average number of adoptions per week is 8, what is P(# adoptions in a week = 5)?
from scipy.stats import poisson
poisson.pmf(5,8)
# P(# adoptions in a week <= 5)?
poisson.cdf(5,8)
# P(# adoptions in a week > 5)?
1 - poisson.cdf(5,8)
```

Central Limit Theorem still applies to poisson distribution.

### Exponential Distribution

Probability of time between Poisson events. Example: Probability of >1 day between adoptions

It uses same lambda value and it is continuous.

![image.png](attachment:d056e5cf-5567-42fd-b61f-43cdd8bae942:image.png)

In Poisson, lambda = 0.5 means 0.5 request per minute. On the other hand, in exponential distribution, 1 / lambda → 1 / 0.5 = 2 which means 1 request per 2 minutes.

```python
from scipy.stats import expon
# Example
# lambda = 0.5 which means scale = 1 / 0.5 = 2
# P(wait < 1 min)
expon.cdf(1, scale = 2)
# P(wait > 4 min)
1 - expon.cdf(4, scale = 2)
```