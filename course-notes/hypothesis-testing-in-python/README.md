# Hypothesis Testing in Python

## Course Overview
This course covers comprehensive hypothesis testing techniques in Python, including parametric and non-parametric tests, effect sizes, and multiple testing corrections.

## Key Topics Covered

### 1. Hypothesis Testing Fundamentals
- Null and alternative hypotheses
- Type I and Type II errors
- P-values and significance levels
- Power analysis

### 2. One-Sample Tests
- One-sample t-test
- One-sample z-test
- Wilcoxon signed-rank test
- Binomial test

### 3. Two-Sample Tests
- Independent t-test
- Paired t-test
- Mann-Whitney U test
- Chi-square tests

### 4. Multiple Testing and ANOVA
- One-way ANOVA
- Two-way ANOVA
- Multiple comparisons
- Bonferroni correction

## Key Concepts

### Hypothesis Testing Framework
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, chi2
import warnings
warnings.filterwarnings('ignore')

def hypothesis_test_framework(data, test_statistic, critical_value=None, p_value=None, 
                            alpha=0.05, test_type='two-tailed'):
    """
    General framework for hypothesis testing.
    
    Args:
        data: Sample data
        test_statistic: Calculated test statistic
        critical_value: Critical value for the test
        p_value: P-value of the test
        alpha: Significance level
        test_type: 'two-tailed', 'left-tailed', or 'right-tailed'
    
    Returns:
        Dictionary with test results
    """
    
    # Decision based on p-value
    if p_value is not None:
        reject_h0_p = p_value < alpha
        decision_p = "Reject H0" if reject_h0_p else "Fail to reject H0"
    else:
        reject_h0_p = None
        decision_p = "Cannot determine (no p-value)"
    
    # Decision based on critical value
    if critical_value is not None:
        if test_type == 'two-tailed':
            reject_h0_critical = abs(test_statistic) > abs(critical_value)
        elif test_type == 'right-tailed':
            reject_h0_critical = test_statistic > critical_value
        elif test_type == 'left-tailed':
            reject_h0_critical = test_statistic < critical_value
        
        decision_critical = "Reject H0" if reject_h0_critical else "Fail to reject H0"
    else:
        reject_h0_critical = None
        decision_critical = "Cannot determine (no critical value)"
    
    return {
        'test_statistic': test_statistic,
        'p_value': p_value,
        'critical_value': critical_value,
        'alpha': alpha,
        'decision_p_value': decision_p,
        'decision_critical': decision_critical,
        'test_type': test_type
    }

def print_test_results(results):
    """Print formatted test results."""
    print("="*50)
    print("HYPOTHESIS TEST RESULTS")
    print("="*50)
    print(f"Test Statistic: {results['test_statistic']:.4f}")
    if results['p_value'] is not None:
        print(f"P-value: {results['p_value']:.4f}")
    if results['critical_value'] is not None:
        print(f"Critical Value: {results['critical_value']:.4f}")
    print(f"Significance Level (α): {results['alpha']}")
    print(f"Test Type: {results['test_type']}")
    print("-"*50)
    print(f"Decision (P-value): {results['decision_p_value']}")
    print(f"Decision (Critical): {results['decision_critical']}")
    print("="*50)
```

### One-Sample Tests
```python
def one_sample_t_test(sample, population_mean, alpha=0.05, alternative='two-sided'):
    """
    Perform one-sample t-test.
    
    Args:
        sample: Sample data
        population_mean: Hypothesized population mean
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with test results
    """
    sample = np.array(sample)
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    # Calculate t-statistic
    t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    
    # Degrees of freedom
    df = n - 1
    
    # Calculate p-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        critical_value = t.ppf(1 - alpha/2, df)
        test_type = 'two-tailed'
    elif alternative == 'less':
        p_value = t.cdf(t_stat, df)
        critical_value = t.ppf(alpha, df)
        test_type = 'left-tailed'
    elif alternative == 'greater':
        p_value = 1 - t.cdf(t_stat, df)
        critical_value = t.ppf(1 - alpha, df)
        test_type = 'right-tailed'
    
    # Calculate effect size (Cohen's d)
    cohens_d = (sample_mean - population_mean) / sample_std
    
    # Using scipy for verification
    scipy_t_stat, scipy_p_value = stats.ttest_1samp(sample, population_mean, alternative=alternative)
    
    results = hypothesis_test_framework(sample, t_stat, critical_value, p_value, alpha, test_type)
    results.update({
        'sample_mean': sample_mean,
        'population_mean': population_mean,
        'sample_std': sample_std,
        'degrees_of_freedom': df,
        'cohens_d': cohens_d,
        'scipy_verification': {'t_stat': scipy_t_stat, 'p_value': scipy_p_value}
    })
    
    return results

def one_sample_z_test(sample, population_mean, population_std, alpha=0.05, alternative='two-sided'):
    """
    Perform one-sample z-test (when population standard deviation is known).
    
    Args:
        sample: Sample data
        population_mean: Hypothesized population mean
        population_std: Known population standard deviation
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with test results
    """
    sample = np.array(sample)
    n = len(sample)
    sample_mean = np.mean(sample)
    
    # Calculate z-statistic
    z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(n))
    
    # Calculate p-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        critical_value = norm.ppf(1 - alpha/2)
        test_type = 'two-tailed'
    elif alternative == 'less':
        p_value = norm.cdf(z_stat)
        critical_value = norm.ppf(alpha)
        test_type = 'left-tailed'
    elif alternative == 'greater':
        p_value = 1 - norm.cdf(z_stat)
        critical_value = norm.ppf(1 - alpha)
        test_type = 'right-tailed'
    
    # Calculate effect size
    cohens_d = (sample_mean - population_mean) / population_std
    
    results = hypothesis_test_framework(sample, z_stat, critical_value, p_value, alpha, test_type)
    results.update({
        'sample_mean': sample_mean,
        'population_mean': population_mean,
        'population_std': population_std,
        'cohens_d': cohens_d
    })
    
    return results

# Example usage
np.random.seed(42)
sample_data = np.random.normal(102, 10, 30)  # Sample with mean close to 102

# One-sample t-test
t_test_results = one_sample_t_test(sample_data, population_mean=100, alpha=0.05)
print_test_results(t_test_results)

# One-sample z-test (assuming known population std)
z_test_results = one_sample_z_test(sample_data, population_mean=100, population_std=10, alpha=0.05)
print_test_results(z_test_results)
```

### Two-Sample Tests
```python
def independent_t_test(sample1, sample2, alpha=0.05, equal_var=True, alternative='two-sided'):
    """
    Perform independent samples t-test.
    
    Args:
        sample1, sample2: Two independent samples
        alpha: Significance level
        equal_var: Assume equal variances (Welch's t-test if False)
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with test results
    """
    sample1, sample2 = np.array(sample1), np.array(sample2)
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    # Calculate t-statistic and degrees of freedom
    if equal_var:
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = np.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test (unequal variances)
        se = np.sqrt(var1/n1 + var2/n2)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    t_stat = (mean1 - mean2) / se
    
    # Calculate p-value and critical value
    if alternative == 'two-sided':
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        critical_value = t.ppf(1 - alpha/2, df)
        test_type = 'two-tailed'
    elif alternative == 'less':
        p_value = t.cdf(t_stat, df)
        critical_value = t.ppf(alpha, df)
        test_type = 'left-tailed'
    elif alternative == 'greater':
        p_value = 1 - t.cdf(t_stat, df)
        critical_value = t.ppf(1 - alpha, df)
        test_type = 'right-tailed'
    
    # Effect size (Cohen's d)
    if equal_var:
        pooled_std = np.sqrt(pooled_var)
        cohens_d = (mean1 - mean2) / pooled_std
    else:
        pooled_std = np.sqrt((var1 + var2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std
    
    # Scipy verification
    scipy_t_stat, scipy_p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var, alternative=alternative)
    
    results = hypothesis_test_framework([sample1, sample2], t_stat, critical_value, p_value, alpha, test_type)
    results.update({
        'mean1': mean1, 'mean2': mean2,
        'var1': var1, 'var2': var2,
        'n1': n1, 'n2': n2,
        'degrees_of_freedom': df,
        'standard_error': se,
        'cohens_d': cohens_d,
        'equal_var_assumed': equal_var,
        'scipy_verification': {'t_stat': scipy_t_stat, 'p_value': scipy_p_value}
    })
    
    return results

def paired_t_test(sample1, sample2, alpha=0.05, alternative='two-sided'):
    """
    Perform paired samples t-test.
    
    Args:
        sample1, sample2: Two paired samples
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with test results
    """
    sample1, sample2 = np.array(sample1), np.array(sample2)
    
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have the same length for paired t-test")
    
    # Calculate differences
    differences = sample1 - sample2
    
    # Perform one-sample t-test on differences
    results = one_sample_t_test(differences, population_mean=0, alpha=alpha, alternative=alternative)
    
    # Add paired-specific information
    results.update({
        'test_type': 'paired t-test',
        'mean1': np.mean(sample1),
        'mean2': np.mean(sample2),
        'mean_difference': np.mean(differences),
        'differences': differences
    })
    
    return results

# Example usage
np.random.seed(42)
group1 = np.random.normal(100, 15, 30)
group2 = np.random.normal(105, 15, 35)

# Independent t-test
independent_results = independent_t_test(group1, group2, alpha=0.05)
print_test_results(independent_results)

# Paired t-test (using same size samples)
group2_paired = np.random.normal(105, 15, 30)
paired_results = paired_t_test(group1, group2_paired, alpha=0.05)
print_test_results(paired_results)
```

### Non-Parametric Tests
```python
def mann_whitney_u_test(sample1, sample2, alpha=0.05, alternative='two-sided'):
    """
    Perform Mann-Whitney U test (non-parametric alternative to independent t-test).
    
    Args:
        sample1, sample2: Two independent samples
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with test results
    """
    # Use scipy for the test
    if alternative == 'two-sided':
        u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
    else:
        u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
    
    # Effect size (rank-biserial correlation)
    n1, n2 = len(sample1), len(sample2)
    r = 1 - (2 * u_stat) / (n1 * n2)  # rank-biserial correlation
    
    results = {
        'test_name': 'Mann-Whitney U Test',
        'u_statistic': u_stat,
        'p_value': p_value,
        'alpha': alpha,
        'alternative': alternative,
        'effect_size_r': r,
        'n1': n1,
        'n2': n2,
        'decision': "Reject H0" if p_value < alpha else "Fail to reject H0"
    }
    
    return results

def wilcoxon_signed_rank_test(sample1, sample2=None, alpha=0.05, alternative='two-sided'):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        sample1: First sample or differences if sample2 is None
        sample2: Second sample (for paired data)
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with test results
    """
    if sample2 is not None:
        # Calculate differences for paired data
        differences = np.array(sample1) - np.array(sample2)
    else:
        # Use sample1 as differences
        differences = np.array(sample1)
    
    # Use scipy for the test
    w_stat, p_value = stats.wilcoxon(differences, alternative=alternative)
    
    # Effect size (r = Z / sqrt(N))
    n = len(differences)
    z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
    r = z_score / np.sqrt(n)
    
    results = {
        'test_name': 'Wilcoxon Signed-Rank Test',
        'w_statistic': w_stat,
        'p_value': p_value,
        'alpha': alpha,
        'alternative': alternative,
        'effect_size_r': r,
        'n': n,
        'decision': "Reject H0" if p_value < alpha else "Fail to reject H0"
    }
    
    return results

# Example usage
np.random.seed(42)
sample_a = np.random.exponential(2, 50)  # Non-normal distribution
sample_b = np.random.exponential(2.5, 55)

# Mann-Whitney U test
mw_results = mann_whitney_u_test(sample_a, sample_b)
print("Mann-Whitney U Test Results:")
for key, value in mw_results.items():
    print(f"{key}: {value}")

# Wilcoxon signed-rank test
sample_b_paired = np.random.exponential(2.5, 50)
wilcoxon_results = wilcoxon_signed_rank_test(sample_a, sample_b_paired)
print("\nWilcoxon Signed-Rank Test Results:")
for key, value in wilcoxon_results.items():
    print(f"{key}: {value}")
```

### Chi-Square Tests
```python
def chi_square_goodness_of_fit(observed, expected=None, alpha=0.05):
    """
    Perform chi-square goodness of fit test.
    
    Args:
        observed: Observed frequencies
        expected: Expected frequencies (uniform if None)
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    observed = np.array(observed)
    
    if expected is None:
        # Assume uniform distribution
        expected = np.full(len(observed), np.sum(observed) / len(observed))
    else:
        expected = np.array(expected)
    
    # Calculate chi-square statistic
    chi2_stat = np.sum((observed - expected)**2 / expected)
    
    # Degrees of freedom
    df = len(observed) - 1
    
    # P-value
    p_value = 1 - chi2.cdf(chi2_stat, df)
    
    # Critical value
    critical_value = chi2.ppf(1 - alpha, df)
    
    # Effect size (Cramér's V)
    n = np.sum(observed)
    cramers_v = np.sqrt(chi2_stat / (n * (min(len(observed), 2) - 1)))
    
    # Scipy verification
    scipy_chi2, scipy_p = stats.chisquare(observed, expected)
    
    results = {
        'test_name': 'Chi-Square Goodness of Fit Test',
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'degrees_of_freedom': df,
        'alpha': alpha,
        'cramers_v': cramers_v,
        'observed': observed,
        'expected': expected,
        'decision': "Reject H0" if p_value < alpha else "Fail to reject H0",
        'scipy_verification': {'chi2_stat': scipy_chi2, 'p_value': scipy_p}
    }
    
    return results

def chi_square_independence(contingency_table, alpha=0.05):
    """
    Perform chi-square test of independence.
    
    Args:
        contingency_table: 2D array or DataFrame with observed frequencies
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    contingency_table = np.array(contingency_table)
    
    # Calculate expected frequencies
    row_totals = np.sum(contingency_table, axis=1)
    col_totals = np.sum(contingency_table, axis=0)
    total = np.sum(contingency_table)
    
    expected = np.outer(row_totals, col_totals) / total
    
    # Calculate chi-square statistic
    chi2_stat = np.sum((contingency_table - expected)**2 / expected)
    
    # Degrees of freedom
    df = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)
    
    # P-value
    p_value = 1 - chi2.cdf(chi2_stat, df)
    
    # Critical value
    critical_value = chi2.ppf(1 - alpha, df)
    
    # Effect size (Cramér's V)
    n = total
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
    
    # Scipy verification
    scipy_chi2, scipy_p, scipy_df, scipy_expected = stats.chi2_contingency(contingency_table)
    
    results = {
        'test_name': 'Chi-Square Test of Independence',
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'degrees_of_freedom': df,
        'alpha': alpha,
        'cramers_v': cramers_v,
        'observed': contingency_table,
        'expected': expected,
        'decision': "Reject H0" if p_value < alpha else "Fail to reject H0",
        'scipy_verification': {'chi2_stat': scipy_chi2, 'p_value': scipy_p, 'df': scipy_df}
    }
    
    return results

# Example usage
# Goodness of fit test
observed_freq = [23, 18, 16, 14, 12, 17]
gof_results = chi_square_goodness_of_fit(observed_freq)
print("Chi-Square Goodness of Fit Results:")
for key, value in gof_results.items():
    if key not in ['observed', 'expected', 'scipy_verification']:
        print(f"{key}: {value}")

# Test of independence
contingency = np.array([[10, 10, 20], [20, 20, 40]])
independence_results = chi_square_independence(contingency)
print("\nChi-Square Independence Test Results:")
for key, value in independence_results.items():
    if key not in ['observed', 'expected', 'scipy_verification']:
        print(f"{key}: {value}")
```

### Power Analysis
```python
def power_analysis_one_sample_t(effect_size, sample_size=None, alpha=0.05, power=None):
    """
    Perform power analysis for one-sample t-test.
    
    Args:
        effect_size: Cohen's d
        sample_size: Sample size (calculate if None)
        alpha: Significance level
        power: Desired power (calculate if None)
    
    Returns:
        Dictionary with power analysis results
    """
    from scipy.stats import t
    
    if sample_size is not None and power is None:
        # Calculate power given sample size
        df = sample_size - 1
        critical_t = t.ppf(1 - alpha/2, df)
        ncp = effect_size * np.sqrt(sample_size)  # Non-centrality parameter
        
        # Power = P(|T| > critical_t | H1 is true)
        power = 1 - t.cdf(critical_t, df, loc=ncp) + t.cdf(-critical_t, df, loc=ncp)
        
        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': power,
            'calculation': 'power'
        }
    
    elif power is not None and sample_size is None:
        # Calculate sample size given power
        # This requires iterative solution
        def power_function(n):
            if n < 2:
                return 0
            df = n - 1
            critical_t = t.ppf(1 - alpha/2, df)
            ncp = effect_size * np.sqrt(n)
            return 1 - t.cdf(critical_t, df, loc=ncp) + t.cdf(-critical_t, df, loc=ncp)
        
        # Find sample size that achieves desired power
        for n in range(2, 1000):
            if power_function(n) >= power:
                sample_size = n
                break
        
        actual_power = power_function(sample_size)
        
        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': actual_power,
            'desired_power': power,
            'calculation': 'sample_size'
        }
    
    else:
        return {"error": "Must specify either sample_size or power, but not both"}

# Example power analysis
print("Power Analysis Examples:")

# Calculate power given sample size
power_result1 = power_analysis_one_sample_t(effect_size=0.5, sample_size=30)
print(f"With effect size 0.5 and n=30: Power = {power_result1['power']:.3f}")

# Calculate sample size given desired power
power_result2 = power_analysis_one_sample_t(effect_size=0.5, power=0.8)
print(f"For power=0.8 and effect size 0.5: Need n = {power_result2['sample_size']}")
```

## Course Notes

# Hypothesis Testing Fundamentals

## A/B Testing

In 2013, EA released SimCity 5. They wanted to increase pre-order of the game. They used A/B testing to test different advertising scenarios. This involves splitting users into control and treatment groups.

```python
# Example
import numpy as np
# Step 3: Repeat steps 1 & 2 many times, appending to a list
so_boot_distn = []
for i in range(5000):
	so_boot_distn.append(
	# Step 2. Calculate point estimate
	np.mean(
			# Step 1. Resample
			stack_overflow.sample(frac=1, replace=True['converted_comp']
```

## Z-Score

standardized value = (value - mean) / standard deviation

z = (sample stat - hypoth. param. value) / standard error

Hypothesis testing use case: Determine whether sample statistics are close to or far away from expected (or hypothesized values)

![image.png](attachment:c6a42ac5-8b9f-413f-94dc-31ee18973ed4:image.png)

```python
# Hypothesize that the proportion is 6%
late_prop_hyp = 0.06

# Calculate the standard error
std_error = np.std(late_shipments_boot_distn, ddof=1)

# Find z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Print z_score
print(z_score)
```

## p-values

**Hypothesis:** A statement about an unknown population parameter.

**Hypothesis Test:** A statistical method that tests two competing hypotheses against each other.

### Two Types of Hypotheses

Null Hypothesis (H₀)

- Represents the existing or current idea/belief
- Often states "no change" or "no difference"
- Assumed to be true at the beginning of the test

Alternative Hypothesis (Hₐ)

- The new "challenger" idea proposed by the researcher
- Claims there is a change or difference from the current situation
- What the researcher is trying to prove

Example Application

**Research Question:** What proportion of data scientists started programming as children?

- **H₀:** The proportion of data scientists who started programming as children is 35%
- **Hₐ:** The proportion of data scientists who started programming as children is greater than 35%

### Fundamental Rules of Hypothesis Testing

1. **Mutual Exclusivity:** Either H₀ or Hₐ is true (not both simultaneously)
2. **Initial Assumption:** H₀ is assumed to be true at the start of the test
3. **Test Outcomes:** The test concludes with one of two decisions:
    - **"Reject H₀"** → Accept Hₐ
    - **"Fail to reject H₀"** → Continue to accept H₀
4. **Decision-Making Criterion:** If the evidence from the sample data significantly supports Hₐ being true, we reject H₀; otherwise, we maintain H₀.

### Significance Level (**α**)

The significance level represents the threshold for "beyond reasonable doubt" in hypothesis testing. Similar to the legal system's requirement for proof "beyond reasonable doubt," we need strong statistical evidence to reject H₀ and accept the alternative hypothesis.

- Common values of α are 0.2, 0.1, 0.05, and 0.01.
- If p ≤ α, reject H₀, else fail to reject H₀.
- α should be set prior to conducting the hypothesis test.

**Key Point:** We never "prove" a hypothesis in statistics; we either have sufficient evidence to reject H₀ or we don't.

![image.png](attachment:902a00cf-6a5a-4832-832a-6335a58df150:image.png)

![image.png](attachment:477083ee-fe48-4858-b20b-e3c5e3d523e1:image.png)

## Calculating the p-value

norm.cdf() is normal CDF from scipy.stats.

- Left-tailed test → use norm.cdf()
- Right-tailed test → user 1-norm.cdf()

```python
from scipy.stats import norm
1 - norm.cdf(z_score, loc = 0, scale = 1)
```

```python
alpha = 0.05
prop_child_samp = (stack_overflow['age_first_code_cut'] == 'child').mean()
pprop_child_hyp = 0.35
std_error = np.std(first_code_boot_distn, ddof = 1)

z_score = (prop_child_samp - prop_child_hyp) / std_error

p_value = 1 - norm.cdf(z_score, loc = 0, scale = 1)
```

### Confidence Intervals

For a significance level of alpha, it’s common to choose a confidence interval level of 1 - alpha.

When α = 0.05 → 95% confidence interval 

|  | actual H0 | actual HA |
| --- | --- | --- |
| chosen H0 | correct | false negative |
| chosen HA | false positive | correct |

False positives are Type 1 errors; false negatives are Type 2 errors.

Possible errors in our example

If p ≤ α, we reject H0:

- A false positive (Type 1) error: data scientists did not start coding as children at a higher rate.

If p ≥ α, we fail to reject H0:

- A false negative (Type 2) error: data scientists started coding as children at a higher rate.

# Two-Sample and ANOVA Tests

## Performing t-tests

![image.png](attachment:22984263-9ced-4040-833e-4348a5427a8c:image.png)

![image.png](attachment:74658f4a-5373-4aaa-b98e-f4288e0cf3fa:image.png)

z = sample stat - population parameter / standard error

t = (difference in sample stats - difference in population parameters) / standard error 

![image.png](attachment:b677eadf-661a-437a-8f48-cf0a06e51d0c:image.png)

```python
# Calculate the numerator of the test statistic
numerator = xbar_no - xbar_yes

# Calculate the denominator of the test statistic
denominator = np.sqrt(s_no ** 2 / n_no + s_yes ** 2 / n_yes)

# Calculate the test statistic
t_stat = numerator / denominator

# Print the test statistic
print(t_stat)
```

## Calculating p-values from t-statistics

### t-distributions

t statistic follows a t-distribution

Have a parameter named degrees of freedom.

![image.png](attachment:670363d6-7445-4fb2-9d9b-b0f6ba8b5a96:image.png)

## Analysis of Variance (ANOVA)

A test for differences between groups. 

```python
alpha = 0.2

pingouin.anova(data=stack_overflow,
								dv='converted_comp',
								between='job_sat')
```

# Proportion Tests

## One-sample proportion tests

p: population proportion (unknown popolation parameter)

p_hat: sample proportion (sample statistic)

p0: hypothesized population proportion

z = (p_hat - p) / SE(p_hat) 

Assuming H0 is true, p = p0 so:

z = (p_hat - p0) / SE(p_hat)

![image.png](attachment:b3958a21-170c-405d-a460-b2fb5b04c228:image.png)

## Two-sample proportion tests

![image.png](attachment:e520125e-58eb-4e2f-a8c5-f5bd6ebbf713:image.png)

![image.png](attachment:67606a47-6396-4a7c-adc5-d7e453a4ba52:image.png)

## Chi-square test of independence

Statistical Independence: Proportion of successes in the response variable is the same across all categories of the explanatory variable.

## Chi-square goodness of fit tests

```python
# Find the number of rows in late_shipments
n_total = len(late_shipments)

# Create n column that is prop column * n_total
hypothesized["n"] = hypothesized["prop"] * n_total

# Plot a red bar graph of n vs. vendor_inco_term for incoterm_counts
plt.bar(incoterm_counts['vendor_inco_term'], incoterm_counts['n'], color="red", label="Observed")

# Add a blue bar plot for the hypothesized counts
plt.bar(hypothesized['vendor_inco_term'], hypothesized['n'],alpha=0.5,color='blue', label="Hypothesized")
plt.legend()
plt.show()
```

# Non-Parametric Tests

## Assumptions in Hypothesis Testing

### Randomness

- Assumption: The samples are random subsets of larger populations. nce: Sace: Sampe
- Consequence: Sample is not representative of population.
- How to check this: Understand how your data was collected.

### Independence of Observations

- Assumption: Each observation (row) in the dataset is independent.
- Consequence: Increased chance of false negative/positive error.
- How to check this: Understand how our data was collected.

### Large Sample Size

- Assumption: The sample is big enough to mitigate uncertainty, so that the Central Limit Theorem applies.
- Consequence: Wider confidence intervals, Increased chance of false negative/positive errors
- How to check this: It depends on the test

### Sanity Check

If the bootstrap distribution does not look normal, assumptions likely aren’t valid. 

Revisit data collection to check for randomness, independence and sample size.

## Non-Parametric Tests

Non-parametric tests avoid the parametric assumptions and conditions. Also, non-parametric tests are more reliable than parametric tests for small sample sizes and when data is not normally distributed.