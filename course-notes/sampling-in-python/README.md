# Sampling in Python

## Course Overview
This course covers various sampling techniques and methods in Python, including random sampling, stratified sampling, bootstrap methods, and sampling distributions.

## Key Topics Covered

### 1. Random Sampling
- Simple random sampling
- Systematic sampling
- Random number generation
- Sampling with and without replacement

### 2. Stratified and Cluster Sampling
- Stratified sampling methods
- Cluster sampling
- Multi-stage sampling
- Quota sampling

### 3. Bootstrap Methods
- Bootstrap sampling
- Bootstrap confidence intervals
- Bootstrap hypothesis testing
- Jackknife methods

### 4. Sampling Distributions
- Central Limit Theorem
- Sampling distribution of means
- Standard error calculation
- Confidence intervals

## Key Concepts

### Simple Random Sampling
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def simple_random_sample(population, sample_size, replace=False):
    """
    Perform simple random sampling from a population.
    
    Args:
        population: List or array of population data
        sample_size: Size of the sample to draw
        replace: Whether to sample with replacement
    
    Returns:
        Array of sampled values
    """
    population = np.array(population)
    
    if replace:
        sample = np.random.choice(population, size=sample_size, replace=True)
    else:
        sample = np.random.choice(population, size=sample_size, replace=False)
    
    return sample

# Example usage
population = np.arange(1, 1001)  # Population of 1000 individuals
sample = simple_random_sample(population, sample_size=100)

print(f"Population mean: {np.mean(population):.2f}")
print(f"Sample mean: {np.mean(sample):.2f}")
print(f"Population std: {np.std(population):.2f}")
print(f"Sample std: {np.std(sample):.2f}")
```

### Systematic Sampling
```python
def systematic_sample(population, sample_size):
    """
    Perform systematic sampling from a population.
    
    Args:
        population: List or array of population data
        sample_size: Desired sample size
    
    Returns:
        Array of systematically sampled values
    """
    population = np.array(population)
    N = len(population)
    k = N // sample_size  # Sampling interval
    
    # Random starting point
    start = np.random.randint(0, k)
    
    # Select every k-th element starting from the random start
    indices = np.arange(start, N, k)[:sample_size]
    sample = population[indices]
    
    return sample, indices

# Example
sys_sample, sys_indices = systematic_sample(population, sample_size=100)
print(f"Systematic sample mean: {np.mean(sys_sample):.2f}")
print(f"Sample indices: {sys_indices[:10]}...")  # Show first 10 indices
```

### Stratified Sampling
```python
def stratified_sample(data, strata_column, sample_size, proportional=True):
    """
    Perform stratified sampling from a DataFrame.
    
    Args:
        data: DataFrame containing the data
        strata_column: Column name to use for stratification
        sample_size: Total sample size desired
        proportional: Whether to use proportional allocation
    
    Returns:
        DataFrame containing the stratified sample
    """
    strata = data[strata_column].unique()
    samples = []
    
    if proportional:
        # Proportional allocation
        for stratum in strata:
            stratum_data = data[data[strata_column] == stratum]
            stratum_size = len(stratum_data)
            stratum_proportion = stratum_size / len(data)
            stratum_sample_size = int(sample_size * stratum_proportion)
            
            if stratum_sample_size > 0:
                stratum_sample = stratum_data.sample(n=stratum_sample_size, replace=False)
                samples.append(stratum_sample)
    else:
        # Equal allocation
        sample_per_stratum = sample_size // len(strata)
        for stratum in strata:
            stratum_data = data[data[strata_column] == stratum]
            stratum_sample = stratum_data.sample(n=min(sample_per_stratum, len(stratum_data)), 
                                               replace=False)
            samples.append(stratum_sample)
    
    return pd.concat(samples, ignore_index=True)

# Example with DataFrame
df = pd.DataFrame({
    'value': np.random.normal(100, 15, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
})

stratified_sample_df = stratified_sample(df, 'category', sample_size=200)
print("Original distribution:")
print(df['category'].value_counts(normalize=True))
print("\nStratified sample distribution:")
print(stratified_sample_df['category'].value_counts(normalize=True))
```

### Bootstrap Sampling
```python
def bootstrap_sample(data, n_bootstrap=1000, statistic=np.mean):
    """
    Perform bootstrap sampling and calculate bootstrap statistics.
    
    Args:
        data: Original sample data
        n_bootstrap: Number of bootstrap samples
        statistic: Function to calculate statistic (e.g., np.mean, np.median)
    
    Returns:
        Array of bootstrap statistics
    """
    data = np.array(data)
    n = len(data)
    bootstrap_stats = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample (sampling with replacement)
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stat = statistic(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    return np.array(bootstrap_stats)

def bootstrap_confidence_interval(data, confidence_level=0.95, n_bootstrap=1000, statistic=np.mean):
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Original sample data
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
        statistic: Function to calculate statistic
    
    Returns:
        Tuple of (lower_bound, upper_bound, bootstrap_stats)
    """
    bootstrap_stats = bootstrap_sample(data, n_bootstrap, statistic)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return lower_bound, upper_bound, bootstrap_stats

# Example usage
sample_data = np.random.normal(50, 10, 100)
lower, upper, bootstrap_means = bootstrap_confidence_interval(sample_data, confidence_level=0.95)

print(f"Original sample mean: {np.mean(sample_data):.2f}")
print(f"Bootstrap 95% CI: ({lower:.2f}, {upper:.2f})")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=50, alpha=0.7, density=True)
plt.axvline(np.mean(sample_data), color='red', linestyle='--', label='Original Sample Mean')
plt.axvline(lower, color='green', linestyle='--', label='95% CI Lower')
plt.axvline(upper, color='green', linestyle='--', label='95% CI Upper')
plt.xlabel('Bootstrap Sample Means')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Sample Means')
plt.legend()
plt.show()
```

### Central Limit Theorem Demonstration
```python
def demonstrate_clt(population, sample_sizes, n_samples=1000):
    """
    Demonstrate the Central Limit Theorem with different sample sizes.
    
    Args:
        population: Population data
        sample_sizes: List of sample sizes to test
        n_samples: Number of samples to draw for each sample size
    
    Returns:
        Dictionary containing sampling distributions
    """
    population = np.array(population)
    results = {}
    
    fig, axes = plt.subplots(2, len(sample_sizes), figsize=(15, 8))
    if len(sample_sizes) == 1:
        axes = axes.reshape(2, 1)
    
    for i, n in enumerate(sample_sizes):
        sample_means = []
        
        # Draw many samples of size n
        for _ in range(n_samples):
            sample = np.random.choice(population, size=n, replace=True)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        results[f'n_{n}'] = sample_means
        
        # Plot original population (top row)
        if i == 0:
            axes[0, i].hist(population, bins=50, alpha=0.7, density=True)
            axes[0, i].set_title('Original Population')
        else:
            axes[0, i].hist(population, bins=50, alpha=0.7, density=True)
            axes[0, i].set_title('Original Population')
        
        # Plot sampling distribution (bottom row)
        axes[1, i].hist(sample_means, bins=50, alpha=0.7, density=True)
        axes[1, i].set_title(f'Sampling Distribution (n={n})')
        axes[1, i].axvline(np.mean(sample_means), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(sample_means):.2f}')
        axes[1, i].legend()
        
        # Calculate theoretical vs empirical standard error
        theoretical_se = np.std(population) / np.sqrt(n)
        empirical_se = np.std(sample_means)
        
        print(f"Sample size {n}:")
        print(f"  Theoretical SE: {theoretical_se:.4f}")
        print(f"  Empirical SE: {empirical_se:.4f}")
        print(f"  Sample mean of means: {np.mean(sample_means):.4f}")
        print()
    
    plt.tight_layout()
    plt.show()
    
    return results

# Example with different distributions
# Uniform distribution
uniform_pop = np.random.uniform(0, 100, 10000)
clt_results_uniform = demonstrate_clt(uniform_pop, sample_sizes=[5, 30, 100])

# Exponential distribution
exponential_pop = np.random.exponential(2, 10000)
clt_results_exp = demonstrate_clt(exponential_pop, sample_sizes=[5, 30, 100])
```

### Sample Size Calculation
```python
def calculate_sample_size_mean(population_std, margin_of_error, confidence_level=0.95):
    """
    Calculate required sample size for estimating population mean.
    
    Args:
        population_std: Population standard deviation (or estimate)
        margin_of_error: Desired margin of error
        confidence_level: Confidence level
    
    Returns:
        Required sample size
    """
    # Get z-score for confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Calculate sample size
    n = (z_score * population_std / margin_of_error) ** 2
    
    return int(np.ceil(n))

def calculate_sample_size_proportion(expected_proportion, margin_of_error, confidence_level=0.95):
    """
    Calculate required sample size for estimating population proportion.
    
    Args:
        expected_proportion: Expected proportion (0-1)
        margin_of_error: Desired margin of error
        confidence_level: Confidence level
    
    Returns:
        Required sample size
    """
    # Get z-score for confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Calculate sample size (worst case if proportion unknown: use 0.5)
    if expected_proportion is None:
        expected_proportion = 0.5
    
    n = (z_score ** 2 * expected_proportion * (1 - expected_proportion)) / (margin_of_error ** 2)
    
    return int(np.ceil(n))

# Examples
print("Sample Size Calculations:")
print(f"For mean (σ=15, E=2, 95% CI): {calculate_sample_size_mean(15, 2, 0.95)}")
print(f"For proportion (p=0.5, E=0.05, 95% CI): {calculate_sample_size_proportion(0.5, 0.05, 0.95)}")
```

### Advanced Sampling Techniques
```python
def cluster_sample(data, cluster_column, n_clusters, cluster_size=None):
    """
    Perform cluster sampling.
    
    Args:
        data: DataFrame containing the data
        cluster_column: Column defining clusters
        n_clusters: Number of clusters to select
        cluster_size: Number of units to sample from each cluster (None = all)
    
    Returns:
        DataFrame containing the cluster sample
    """
    # Get unique clusters
    clusters = data[cluster_column].unique()
    
    # Randomly select clusters
    selected_clusters = np.random.choice(clusters, size=n_clusters, replace=False)
    
    samples = []
    for cluster in selected_clusters:
        cluster_data = data[data[cluster_column] == cluster]
        
        if cluster_size is None or cluster_size >= len(cluster_data):
            # Take all units in the cluster
            samples.append(cluster_data)
        else:
            # Sample from the cluster
            cluster_sample = cluster_data.sample(n=cluster_size, replace=False)
            samples.append(cluster_sample)
    
    return pd.concat(samples, ignore_index=True)

def multistage_sample(data, stage1_column, stage2_column, n_stage1, n_stage2_per_stage1):
    """
    Perform two-stage sampling.
    
    Args:
        data: DataFrame containing the data
        stage1_column: Column for first stage sampling units
        stage2_column: Column for second stage sampling units
        n_stage1: Number of first stage units to select
        n_stage2_per_stage1: Number of second stage units per first stage unit
    
    Returns:
        DataFrame containing the multistage sample
    """
    # Stage 1: Select primary sampling units
    stage1_units = data[stage1_column].unique()
    selected_stage1 = np.random.choice(stage1_units, size=n_stage1, replace=False)
    
    samples = []
    for stage1_unit in selected_stage1:
        # Get data for this stage 1 unit
        stage1_data = data[data[stage1_column] == stage1_unit]
        
        # Stage 2: Select secondary sampling units within this stage 1 unit
        stage2_units = stage1_data[stage2_column].unique()
        n_available = len(stage2_units)
        n_to_select = min(n_stage2_per_stage1, n_available)
        
        selected_stage2 = np.random.choice(stage2_units, size=n_to_select, replace=False)
        
        # Get all observations for selected stage 2 units
        for stage2_unit in selected_stage2:
            stage2_data = stage1_data[stage1_data[stage2_column] == stage2_unit]
            samples.append(stage2_data)
    
    return pd.concat(samples, ignore_index=True)

# Example multistage sampling
df_multistage = pd.DataFrame({
    'region': np.repeat(['North', 'South', 'East', 'West'], 250),
    'city': np.tile(np.repeat(['City_' + str(i) for i in range(1, 21)], 50), 1),
    'household': np.tile(range(1, 51), 20),
    'income': np.random.normal(50000, 15000, 1000)
})

multistage_sample_result = multistage_sample(df_multistage, 'region', 'city', n_stage1=2, n_stage2_per_stage1=3)
print(f"Original data shape: {df_multistage.shape}")
print(f"Multistage sample shape: {multistage_sample_result.shape}")
print(f"Selected regions: {multistage_sample_result['region'].unique()}")
print(f"Selected cities: {multistage_sample_result['city'].unique()}")
```

## Course Notes

# Introduction to Sampling

## Sampling and Point Estimates

pandas .sample(n) method returns n number of random rows.

```python
cup_points_samp = coffee_ratings['total_cup_points'].sample(n=10)
```

A point estimate or sample statistic is a calculation made on the sample dataset.

```python
# Sample 1000 rows from spotify_population
spotify_sample = spotify_population.sample(n=1000)

# Print the sample
print(spotify_sample)
```

## Convenience Sampling

Sample not representative of population, causing simple bias. 

Collecting data by the easiest method is called convenience sampling.

```python
# Update the histogram to use spotify_mysterious_sample2
spotify_mysterious_sample2['duration_minutes'].hist(bins=np.arange(0, 15.5, 0.5))
plt.show()
```

## Pseudo-random number generation

```python
# Example
# Generate random numbers from a Uniform(-3, 3)
uniforms = np.random.uniform(-3,3,5000)

# Print uniforms
print(uniforms)
```

# Sampling Methods

## Simple Random and Systematic Random

```python
# Defining the interval
sample_size = 5
pop_size = len(coffee_ratings)

interval = pop_size // sample_size

coffee_ratings.iloc[::interval]
```

```python
# Example
# Set the sample size to 70
sample_size = 70

# Calculate the population size from attrition_pop
pop_size = len(attrition_pop)

# Calculate the interval
interval = pop_size // sample_size

# Systematically sample 70 rows
attrition_sys_samp = attrition_pop.iloc[::interval]

# Print the sample
print(attrition_sys_samp)
```

## Stratified and Weighted Random Sampling

```python
top_counted_countries = ['Mexico','Columbia', 'Guatemala', 'Brazil', 'Taiwan', 'United States (Hawaii)']
top_counted_subset = coffee_ratings['country_of_origin'].isin(top_counted_countries)

coffee_ratings_top = coffee_ratings['top_counted_subset']

coffee_ratings_samp = coffee_ratings_top.sample(frac = 0.1, random_state=2021)

coffee_ratings_strat = coffee_ratings_top.groupby('country_of_origin').sample(frac=0.1, random_state=2021)

# With groupby method our sample ratios will be close to the actual population ratios
```

```python
# Equal Counts stratified sampling
coffee_ratings_eq = coffee_ratings_top.groupby('country_of_origin').sample(n=15, random_state=2021)

```

In weighted random sampling, user should specify weights to adjust the relative probability of a row being sampled.

```python
# Example
import numpy as np
coffee_ratings_weight = coffee_ratings_top
condition = coffee_ratings_weight['country_of_origin'] == 'Taiwan'

coffee_ratings_weight['weight'] = np.where(condition,2,1)

coffee_ratings_weight = coffee_ratings_weight.sample(frac=0.1, weights='weight')

```

```python
# Example
# Proportion of employees by Education level
education_counts_pop = attrition_pop['Education'].value_counts(normalize=True)

# Print education_counts_pop
print(education_counts_pop)

# Proportional stratified sampling for 40% of each Education group
attrition_strat = attrition_pop.groupby('Education')\
	.sample(frac=0.4, random_state=2022)

# Calculate the Education level proportions from attrition_strat
education_counts_strat = attrition_strat['Education'].value_counts(normalize=True)

# Print education_counts_strat
print(education_counts_strat)
```

## Cluster sampling

Stratified sampling vs. cluster sampling

In stratified sampling:

- split the population into subgroups
- Use simple random sampling on every subgroup

In cluster sampling:

- Use simple random sampling to pick some subgroups
- Use simple random sampling on only those subgroups

```python
import random
varieties_samp = random.sample(varieties_pop, k = 3)

variety_condition = coffee_ratings['variety'].isin(varieties_samp)
coffee_ratings_cluster = coffee_ratings[variety_condition]

coffee_ratings_cluster['variety'] = coffee_ratings_cluster['variety'].cat.remove_unused_categories()

coffee_ratings_cluster.groupby('variety').sample(n = 5, random_state = 2021)
```

Cluster sampling is a type of multistage sampling. Can have > 2 stages.

```python
# Example
# Create a list of unique JobRole values
job_roles_pop = list(attrition_pop['JobRole'].unique())

# Randomly sample four JobRole values
job_roles_samp = random.sample(job_roles_pop, k=4)

# Filter for rows where JobRole is in job_roles_samp
jobrole_condition = attrition_pop['JobRole'].isin(job_roles_samp)
attrition_filtered = attrition_pop[jobrole_condition]

# Remove categories with no rows
attrition_filtered['JobRole'] = attrition_filtered['JobRole'].cat.remove_unused_categories()

# Randomly sample 10 employees from each sampled job role
attrition_clust = attrition_filtered.groupby('JobRole').sample(n=10, random_state=2022)

# Print the sample
print(attrition_clust)
```

# Sampling Distributions

## Relative Error of Point Estimates

Larger sample size will give more accurate results

```python
# Population parameter
population_mean = coffee_ratings['total_cup_points'].mean()

# Point estimate
sample_mean = coffee_ratings.sample(n = sample_size)['total_cup_points'].mean()

# Relative error
rel_error_pct = 100 * abs(population_mean-sample_mean) / population_mean
```

## Creating a sampling distribution

```python
mean_cup_points_1000 = []

for i in range(1000):
		mean_cup_points_1000.append(
				coffee_ratings.sample(n=30)['total_cup_points'].mean())
```

A sampling distribution is a distribution of replicates of point estimates.

## Approximate sampling distribution

```python
# Expand a grid representing 5 8-sided dice
dice = expand_grid(
  {'die1': [1, 2, 3, 4, 5, 6, 7, 8],
   'die2': [1, 2, 3, 4, 5, 6, 7, 8],
   'die3': [1, 2, 3, 4, 5, 6, 7, 8],
   'die4': [1, 2, 3, 4, 5, 6, 7, 8],
   'die5': [1, 2, 3, 4, 5, 6, 7, 8]
  })

# Add a column of mean rolls and convert to a categorical
dice['mean_roll'] = (dice['die1'] + dice['die2'] + 
                     dice['die3'] + dice['die4'] + 
                     dice['die5']) / 5
dice['mean_roll'] = dice['mean_roll'].astype('category')

# Draw a bar plot of mean_roll
dice['mean_roll'].value_counts(sort=False).plot(kind='bar')
plt.show()
```

## Standard errors and the Central Limit Theorem

Averages of independent samples have approximately normal distributions. As the sample size increases, the distribution of the averages gets closer to being normally distributed. Furthermore, the width of the sampling distribution gets narrower.

![image.png](attachment:5b8fd4e2-1003-4a3b-a5ca-588a21c9e9d3:image.png)

```python
coffee_ratings['total_cup_points'].std(ddof = 0)
```

Specify ddof = 0 when calling .std() on populations. Specify, ddof = 1 when calling np.std() on samples or sampling distributions.

Standard error is standard deviation of the sampling distribution. Important tool in understanding sampling variability.

# Bootstrap Distributions

## Introduction to Boostrapping

Bootstrapping is the opposite of sampling from a population.

Sampling: going from a population to a smaller sample

Bootstrapping: building up a theoretical population from the sample

### Bootstrapping process

1. Make a resample of the same size as the original sample
2. Calculate the statistic of interest for this bootstrap sample
3. Repeat steps 1 and 2 many times

```python
import numpy as np
mean_flavors_1000 = []
for i in range(1000):
		mean_flavors_1000.append(
			np.mean(coffee_sample.sample(frac = 1, replace = True)['flavor'])
			)
```

Bootstrap distribution mean is usually close to the sample mean. However, may not be a good estimate of the population mean. Bootstrapping cannot correct biases from sampling.

In standard deviation, standard error times square root of sample size estimates the population standard deviation. 

```python
standard_error * np.sqrt(500)
```

![image.png](attachment:7c1b8c7e-56cb-418f-8abd-e6b4224e8df9:image.png)

## Confidence Intervals

Values within one standard deviation of the mean includes a large number of values from each of these distributions.

![image.png](attachment:dd1ee714-f3db-4e16-8351-0cbfd91154c0:image.png)

![image.png](attachment:769569f3-6c97-4aa0-bf32-0d879d316b86:image.png)

![image.png](attachment:743d05f9-8aed-4aa5-a418-8f9215fd432c:image.png)