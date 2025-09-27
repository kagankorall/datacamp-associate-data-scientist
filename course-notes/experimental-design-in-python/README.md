# Experimental Design in Python

## Course Overview
This course covers principles of experimental design, including randomization, control groups, blocking, factorial designs, and analyzing experimental data using Python.

## Key Topics Covered

### 1. Fundamentals of Experimental Design
- Controlled experiments vs observational studies
- Randomization and control groups
- Confounding variables and bias
- Causality vs correlation

### 2. Types of Experimental Designs
- Completely randomized design
- Randomized block design
- Factorial design
- Repeated measures design

### 3. Sample Size and Power
- Power analysis for experiments
- Effect size estimation
- Multiple testing corrections
- Practical significance vs statistical significance

### 4. A/B Testing and Online Experiments
- A/B test design and analysis
- Sequential testing
- Multi-armed bandit problems
- Conversion rate optimization

## Key Concepts

### Experimental Design Principles
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t
import itertools
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_experimental_data(n_per_group, treatment_effects, noise_std=1, seed=42):
    """
    Generate simulated experimental data.
    
    Args:
        n_per_group: Number of observations per group
        treatment_effects: List of treatment effects (control = 0)
        noise_std: Standard deviation of random noise
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with experimental data
    """
    np.random.seed(seed)
    
    data = []
    for i, effect in enumerate(treatment_effects):
        group_data = {
            'group': f'Group_{i}',
            'treatment': i,
            'outcome': np.random.normal(effect, noise_std, n_per_group)
        }
        group_df = pd.DataFrame(group_data)
        data.append(group_df)
    
    return pd.concat(data, ignore_index=True)

def randomize_participants(participants, n_groups, method='simple', block_variable=None):
    """
    Randomize participants to experimental groups.
    
    Args:
        participants: DataFrame with participant information
        n_groups: Number of experimental groups
        method: 'simple', 'block', or 'stratified'
        block_variable: Variable to use for blocking (if method='block')
    
    Returns:
        DataFrame with group assignments
    """
    participants = participants.copy()
    n_participants = len(participants)
    
    if method == 'simple':
        # Simple randomization
        groups = np.random.choice(range(n_groups), size=n_participants, replace=True)
        participants['assigned_group'] = groups
        
    elif method == 'block' and block_variable is not None:
        # Blocked randomization
        participants['assigned_group'] = -1
        
        for block_value in participants[block_variable].unique():
            block_mask = participants[block_variable] == block_value
            block_participants = participants[block_mask]
            n_block = len(block_participants)
            
            # Ensure equal allocation within blocks
            groups_per_block = np.repeat(range(n_groups), n_block // n_groups)
            if n_block % n_groups != 0:
                groups_per_block = np.append(groups_per_block, 
                                           np.random.choice(range(n_groups), 
                                                          n_block % n_groups, 
                                                          replace=False))
            
            np.random.shuffle(groups_per_block)
            participants.loc[block_mask, 'assigned_group'] = groups_per_block
    
    elif method == 'stratified' and block_variable is not None:
        # Stratified randomization (similar to block but ensures proportional allocation)
        participants['assigned_group'] = -1
        
        for stratum_value in participants[block_variable].unique():
            stratum_mask = participants[block_variable] == stratum_value
            stratum_participants = participants[stratum_mask]
            n_stratum = len(stratum_participants)
            
            # Proportional allocation
            groups = np.random.choice(range(n_groups), size=n_stratum, replace=True)
            participants.loc[stratum_mask, 'assigned_group'] = groups
    
    return participants

# Example usage
participants_df = pd.DataFrame({
    'participant_id': range(1, 101),
    'age_group': np.random.choice(['Young', 'Middle', 'Old'], 100),
    'gender': np.random.choice(['M', 'F'], 100),
    'baseline_score': np.random.normal(50, 10, 100)
})

# Simple randomization
simple_randomized = randomize_participants(participants_df, n_groups=3, method='simple')
print("Simple Randomization - Group Distribution:")
print(simple_randomized['assigned_group'].value_counts())

# Block randomization by age group
block_randomized = randomize_participants(participants_df, n_groups=3, 
                                        method='block', block_variable='age_group')
print("\nBlock Randomization by Age Group:")
print(pd.crosstab(block_randomized['age_group'], block_randomized['assigned_group']))
```

### A/B Testing Framework
```python
class ABTest:
    """
    A/B Testing framework for experimental analysis.
    """
    
    def __init__(self, control_data, treatment_data, metric='conversion_rate'):
        """
        Initialize A/B test with control and treatment data.
        
        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            metric: Type of metric ('conversion_rate', 'continuous')
        """
        self.control_data = np.array(control_data)
        self.treatment_data = np.array(treatment_data)
        self.metric = metric
        
        # Calculate basic statistics
        self.n_control = len(self.control_data)
        self.n_treatment = len(self.treatment_data)
        
        if metric == 'conversion_rate':
            self.control_conversions = np.sum(self.control_data)
            self.treatment_conversions = np.sum(self.treatment_data)
            self.control_rate = self.control_conversions / self.n_control
            self.treatment_rate = self.treatment_conversions / self.n_treatment
        else:
            self.control_mean = np.mean(self.control_data)
            self.treatment_mean = np.mean(self.treatment_data)
            self.control_std = np.std(self.control_data, ddof=1)
            self.treatment_std = np.std(self.treatment_data, ddof=1)
    
    def calculate_effect_size(self):
        """Calculate effect size and confidence interval."""
        if self.metric == 'conversion_rate':
            # Relative lift
            relative_lift = (self.treatment_rate - self.control_rate) / self.control_rate
            
            # Standard error for difference in proportions
            se = np.sqrt(self.control_rate * (1 - self.control_rate) / self.n_control + 
                        self.treatment_rate * (1 - self.treatment_rate) / self.n_treatment)
            
            return {
                'absolute_difference': self.treatment_rate - self.control_rate,
                'relative_lift': relative_lift,
                'standard_error': se
            }
        else:
            # Cohen's d for continuous metrics
            pooled_std = np.sqrt(((self.n_control - 1) * self.control_std**2 + 
                                (self.n_treatment - 1) * self.treatment_std**2) / 
                               (self.n_control + self.n_treatment - 2))
            
            cohens_d = (self.treatment_mean - self.control_mean) / pooled_std
            
            # Standard error for difference in means
            se = np.sqrt(self.control_std**2 / self.n_control + 
                        self.treatment_std**2 / self.n_treatment)
            
            return {
                'mean_difference': self.treatment_mean - self.control_mean,
                'cohens_d': cohens_d,
                'standard_error': se
            }
    
    def statistical_test(self, alpha=0.05, alternative='two-sided'):
        """Perform statistical test."""
        if self.metric == 'conversion_rate':
            # Two-proportion z-test
            p_pool = (self.control_conversions + self.treatment_conversions) / (self.n_control + self.n_treatment)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/self.n_control + 1/self.n_treatment))
            
            z_stat = (self.treatment_rate - self.control_rate) / se_pool
            
            if alternative == 'two-sided':
                p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            elif alternative == 'greater':
                p_value = 1 - norm.cdf(z_stat)
            elif alternative == 'less':
                p_value = norm.cdf(z_stat)
            
            return {
                'test_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'test_type': 'two-proportion z-test'
            }
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(self.treatment_data, self.control_data, 
                                            alternative=alternative)
            
            return {
                'test_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'test_type': 'independent t-test'
            }
    
    def confidence_interval(self, confidence_level=0.95):
        """Calculate confidence interval for the effect."""
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha/2)
        
        effect_size = self.calculate_effect_size()
        se = effect_size['standard_error']
        
        if self.metric == 'conversion_rate':
            diff = effect_size['absolute_difference']
            margin_error = z_score * se
            return {
                'lower_bound': diff - margin_error,
                'upper_bound': diff + margin_error,
                'estimate': diff
            }
        else:
            diff = effect_size['mean_difference']
            # Use t-distribution for small samples
            df = self.n_control + self.n_treatment - 2
            t_score = t.ppf(1 - alpha/2, df)
            margin_error = t_score * se
            return {
                'lower_bound': diff - margin_error,
                'upper_bound': diff + margin_error,
                'estimate': diff
            }
    
    def power_analysis(self, effect_size, alpha=0.05, power=0.8):
        """Calculate required sample size for given power."""
        if self.metric == 'conversion_rate':
            # For proportions
            p1 = self.control_rate
            p2 = p1 + effect_size
            p_avg = (p1 + p2) / 2
            
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            
            n_per_group = ((z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) + 
                           z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / 
                          (p2 - p1))**2
        else:
            # For continuous variables (Cohen's d)
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            
            n_per_group = 2 * ((z_alpha + z_beta) / effect_size)**2
        
        return int(np.ceil(n_per_group))
    
    def generate_report(self):
        """Generate comprehensive A/B test report."""
        effect_size = self.calculate_effect_size()
        statistical_test = self.statistical_test()
        ci = self.confidence_interval()
        
        print("="*60)
        print("A/B TEST RESULTS REPORT")
        print("="*60)
        
        if self.metric == 'conversion_rate':
            print(f"Control Group: {self.n_control} participants, {self.control_conversions} conversions")
            print(f"Control Rate: {self.control_rate:.4f} ({self.control_rate*100:.2f}%)")
            print(f"Treatment Group: {self.n_treatment} participants, {self.treatment_conversions} conversions")
            print(f"Treatment Rate: {self.treatment_rate:.4f} ({self.treatment_rate*100:.2f}%)")
            print(f"Absolute Difference: {effect_size['absolute_difference']:.4f}")
            print(f"Relative Lift: {effect_size['relative_lift']*100:.2f}%")
        else:
            print(f"Control Group: {self.n_control} participants, Mean = {self.control_mean:.4f}")
            print(f"Treatment Group: {self.n_treatment} participants, Mean = {self.treatment_mean:.4f}")
            print(f"Mean Difference: {effect_size['mean_difference']:.4f}")
            print(f"Cohen's d: {effect_size['cohens_d']:.4f}")
        
        print(f"\nStatistical Test: {statistical_test['test_type']}")
        print(f"Test Statistic: {statistical_test['test_statistic']:.4f}")
        print(f"P-value: {statistical_test['p_value']:.4f}")
        print(f"Significant: {statistical_test['significant']}")
        
        print(f"\n95% Confidence Interval: ({ci['lower_bound']:.4f}, {ci['upper_bound']:.4f})")
        
        print("="*60)

# Example A/B test
np.random.seed(42)

# Conversion rate test
control_conversions = np.random.binomial(1, 0.10, 1000)  # 10% baseline conversion
treatment_conversions = np.random.binomial(1, 0.12, 1000)  # 12% treatment conversion

ab_test = ABTest(control_conversions, treatment_conversions, metric='conversion_rate')
ab_test.generate_report()

# Continuous metric test
control_continuous = np.random.normal(100, 15, 500)
treatment_continuous = np.random.normal(105, 15, 500)

ab_test_continuous = ABTest(control_continuous, treatment_continuous, metric='continuous')
ab_test_continuous.generate_report()
```

### Factorial Design Analysis
```python
def factorial_design_analysis(data, factors, response, interactions=True):
    """
    Analyze factorial design experiment.
    
    Args:
        data: DataFrame with experimental data
        factors: List of factor column names
        response: Response variable column name
        interactions: Whether to include interaction effects
    
    Returns:
        Dictionary with analysis results
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    # Create formula for ANOVA
    if interactions and len(factors) > 1:
        # Include all interactions
        interaction_terms = []
        for r in range(2, len(factors) + 1):
            for combo in itertools.combinations(factors, r):
                interaction_terms.append(':'.join(combo))
        
        formula = f"{response} ~ {' + '.join(factors)} + {' + '.join(interaction_terms)}"
    else:
        formula = f"{response} ~ {' + '.join(factors)}"
    
    # Fit ANOVA model
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Calculate effect sizes (eta-squared)
    ss_total = anova_table['sum_sq'].sum()
    anova_table['eta_squared'] = anova_table['sum_sq'] / ss_total
    
    # Main effects analysis
    main_effects = {}
    for factor in factors:
        factor_means = data.groupby(factor)[response].agg(['mean', 'std', 'count'])
        main_effects[factor] = factor_means
    
    # Interaction plots if applicable
    if len(factors) >= 2:
        fig, axes = plt.subplots(1, len(factors)-1, figsize=(15, 5))
        if len(factors) == 2:
            axes = [axes]
        
        for i, (factor1, factor2) in enumerate(itertools.combinations(factors, 2)):
            if i < len(axes):
                interaction_plot = data.groupby([factor1, factor2])[response].mean().unstack()
                interaction_plot.plot(kind='line', ax=axes[i], marker='o')
                axes[i].set_title(f'Interaction: {factor1} × {factor2}')
                axes[i].set_ylabel(f'Mean {response}')
                axes[i].legend(title=factor2)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'anova_table': anova_table,
        'model': model,
        'main_effects': main_effects,
        'formula': formula
    }

# Example factorial design
np.random.seed(42)

# 2x2 factorial design: Factor A (Method) x Factor B (Time)
n_per_cell = 25
factorial_data = []

for method in ['Method1', 'Method2']:
    for time in ['Morning', 'Evening']:
        # Different effects for each combination
        if method == 'Method1' and time == 'Morning':
            scores = np.random.normal(80, 10, n_per_cell)
        elif method == 'Method1' and time == 'Evening':
            scores = np.random.normal(75, 10, n_per_cell)
        elif method == 'Method2' and time == 'Morning':
            scores = np.random.normal(85, 10, n_per_cell)
        else:  # Method2, Evening
            scores = np.random.normal(90, 10, n_per_cell)
        
        for score in scores:
            factorial_data.append({
                'Method': method,
                'Time': time,
                'Score': score
            })

factorial_df = pd.DataFrame(factorial_data)

# Analyze factorial design
factorial_results = factorial_design_analysis(factorial_df, 
                                            factors=['Method', 'Time'], 
                                            response='Score')

print("Factorial Design ANOVA Results:")
print(factorial_results['anova_table'])

print("\nMain Effects - Method:")
print(factorial_results['main_effects']['Method'])

print("\nMain Effects - Time:")
print(factorial_results['main_effects']['Time'])
```

### Power Analysis for Experimental Design
```python
def experimental_power_analysis(effect_size, n_groups=2, alpha=0.05, power=None, n_per_group=None):
    """
    Power analysis for experimental designs.
    
    Args:
        effect_size: Expected effect size (Cohen's f for ANOVA, Cohen's d for t-test)
        n_groups: Number of groups in the experiment
        alpha: Significance level
        power: Desired power (calculate n if provided)
        n_per_group: Sample size per group (calculate power if provided)
    
    Returns:
        Dictionary with power analysis results
    """
    
    if n_groups == 2:
        # Two-group comparison (t-test)
        if power is not None and n_per_group is None:
            # Calculate required sample size
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            n_required = 2 * ((z_alpha + z_beta) / effect_size)**2
            
            return {
                'test_type': 'Two-group t-test',
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'n_per_group_required': int(np.ceil(n_required)),
                'total_n_required': int(np.ceil(n_required * 2))
            }
        
        elif n_per_group is not None and power is None:
            # Calculate achieved power
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(n_per_group/2) - z_alpha
            achieved_power = norm.cdf(z_beta)
            
            return {
                'test_type': 'Two-group t-test',
                'effect_size': effect_size,
                'alpha': alpha,
                'n_per_group': n_per_group,
                'total_n': n_per_group * 2,
                'achieved_power': achieved_power
            }
    
    else:
        # Multi-group comparison (ANOVA)
        # Convert Cohen's f to Cohen's d equivalent
        cohens_d_equiv = effect_size * np.sqrt(2)
        
        if power is not None and n_per_group is None:
            # Approximate calculation for ANOVA
            df_error = n_groups * (n_per_group - 1) if n_per_group else 30  # Assume initial
            df_treatment = n_groups - 1
            
            # Iterative solution for sample size
            for n in range(5, 500):
                df_error = n_groups * (n - 1)
                ncp = effect_size**2 * n * n_groups  # Non-centrality parameter
                
                # Critical F-value
                f_critical = stats.f.ppf(1 - alpha, df_treatment, df_error)
                
                # Power calculation
                power_achieved = 1 - stats.ncf.cdf(f_critical, df_treatment, df_error, ncp)
                
                if power_achieved >= power:
                    n_required = n
                    break
            
            return {
                'test_type': f'{n_groups}-group ANOVA',
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'n_per_group_required': n_required,
                'total_n_required': n_required * n_groups,
                'df_treatment': df_treatment,
                'df_error': n_groups * (n_required - 1)
            }
        
        elif n_per_group is not None and power is None:
            # Calculate achieved power for ANOVA
            df_treatment = n_groups - 1
            df_error = n_groups * (n_per_group - 1)
            ncp = effect_size**2 * n_per_group * n_groups
            
            f_critical = stats.f.ppf(1 - alpha, df_treatment, df_error)
            achieved_power = 1 - stats.ncf.cdf(f_critical, df_treatment, df_error, ncp)
            
            return {
                'test_type': f'{n_groups}-group ANOVA',
                'effect_size': effect_size,
                'alpha': alpha,
                'n_per_group': n_per_group,
                'total_n': n_per_group * n_groups,
                'achieved_power': achieved_power,
                'df_treatment': df_treatment,
                'df_error': df_error
            }

# Example power analyses
print("Power Analysis Examples:")

# Two-group comparison
power_result1 = experimental_power_analysis(effect_size=0.5, n_groups=2, power=0.8)
print(f"Two-group test: Need {power_result1['n_per_group_required']} per group for 80% power")

# Multi-group comparison
power_result2 = experimental_power_analysis(effect_size=0.25, n_groups=4, power=0.8)
print(f"Four-group ANOVA: Need {power_result2['n_per_group_required']} per group for 80% power")

# Check achieved power
power_result3 = experimental_power_analysis(effect_size=0.5, n_groups=2, n_per_group=30)
print(f"With n=30 per group: Achieved power = {power_result3['achieved_power']:.3f}")
```

### Multiple Testing Corrections
```python
def multiple_testing_correction(p_values, method='bonferroni', alpha=0.05):
    """
    Apply multiple testing corrections.
    
    Args:
        p_values: List of p-values
        method: 'bonferroni', 'holm', 'benjamini_hochberg', or 'benjamini_yekutieli'
        alpha: Family-wise error rate
    
    Returns:
        Dictionary with corrected results
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)
    
    if method == 'bonferroni':
        # Bonferroni correction
        corrected_alpha = alpha / n_tests
        adjusted_p = p_values * n_tests
        adjusted_p = np.minimum(adjusted_p, 1.0)  # Cap at 1.0
        
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        adjusted_p = np.zeros_like(p_values)
        corrected_alpha = alpha / n_tests
        
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            step_alpha = alpha / (n_tests - i)
            adjusted_p[idx] = p * (n_tests - i)
            adjusted_p[idx] = min(adjusted_p[idx], 1.0)
    
    elif method == 'benjamini_hochberg':
        # Benjamini-Hochberg (FDR control)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        adjusted_p = np.zeros_like(p_values)
        
        for i in range(n_tests-1, -1, -1):
            idx = sorted_indices[i]
            rank = i + 1
            adjusted_p[idx] = sorted_p[i] * n_tests / rank
            
            if i < n_tests - 1:
                # Ensure monotonicity
                next_idx = sorted_indices[i + 1]
                adjusted_p[idx] = min(adjusted_p[idx], adjusted_p[next_idx])
        
        adjusted_p = np.minimum(adjusted_p, 1.0)
        corrected_alpha = alpha  # For FDR, we still use original alpha
    
    # Determine significance
    if method in ['bonferroni', 'holm']:
        significant = adjusted_p < alpha
    else:  # benjamini_hochberg
        significant = adjusted_p < alpha
    
    return {
        'original_p_values': p_values,
        'adjusted_p_values': adjusted_p,
        'method': method,
        'alpha': alpha,
        'significant': significant,
        'n_significant': np.sum(significant),
        'n_tests': n_tests
    }

# Example multiple testing
np.random.seed(42)
# Generate some p-values (mix of significant and non-significant)
example_p_values = np.concatenate([
    np.random.uniform(0, 0.01, 3),    # 3 truly significant
    np.random.uniform(0.01, 0.05, 2), # 2 marginally significant  
    np.random.uniform(0.05, 1.0, 15)  # 15 non-significant
])

print("Multiple Testing Correction Example:")
print(f"Original p-values: {example_p_values[:10]}...")  # Show first 10

# Apply different corrections
bonferroni_results = multiple_testing_correction(example_p_values, method='bonferroni')
holm_results = multiple_testing_correction(example_p_values, method='holm')
bh_results = multiple_testing_correction(example_p_values, method='benjamini_hochberg')

print(f"\nBonferroni: {bonferroni_results['n_significant']} significant out of {bonferroni_results['n_tests']}")
print(f"Holm: {holm_results['n_significant']} significant out of {holm_results['n_tests']}")
print(f"Benjamini-Hochberg: {bh_results['n_significant']} significant out of {bh_results['n_tests']}")
```

## Course Notes
# Experimental Design Preliminaries

## Setting up Experiments

How to assign subjects to groups?

1. Option: ron-random (split the DataFrame)
2. Option: random assignment

### Non-random Assignment

```python
group1_nonrandom = heights.iloc[0:100,:]
group2_nonrandom = heights.iloc[100:,:]
compare_df = pd.concat(
			[group1_nonrandom['height'].describe(),
			 group2_nonrandom['height'].describe()],
			 axis=1)
compare_df.columns = ['group1', 'group2']
```

```python
# Example
# Non-random assignment
group1_non_rand = weights.iloc[0:250,:]
group2_non_rand = weights.iloc[250:,:]

# Compare descriptive statistics of groups
compare_df_non_rand = pd.concat([group1_non_rand['weight'].describe(), group2_non_rand['weight'].describe()], axis=1)
compare_df_non_rand.columns = ['group1', 'group2']

# Print to assess
print(compare_df_non_rand)
```

### Random Assignment

```python
group1 = heights.sample(frac=0.5, replace=False, random_state=42)
group2 = heigths.drop(group1.index)
```

```python
# Example
# Randomly assign half
group1_random = weights.sample(n=250, random_state=42, replace=False)

# Create second assignment
group2_random = weights.drop(group1_random.index)

# Compare assignments
compare_df_random = pd.concat([group1_random['weight'].describe(), group2_random['weight'].describe()], axis=1)
compare_df_random.columns = ['group1', 'group2']
print(compare_df_random)
```

## Experimental Data Setup

### The Problem with Randomization

1. Uneven issue: Different number of subjects in groups
2. Covariate issue: High variability in some covariates → group imbalances in randomization. Result: Harder to measure treatment effect

### Block Randomization

Split into blocks of size n first then randomly split. This fixes uneven issue.

```python
group1 = ecom.sample(frac=0.5, random_state=42, replace=False)
group1['Block'] = 1
group2 = ecom.drop(group1.index)
group2['Block'] = 2

# Visualizing splits
sns.displot(data = ecom,
						x = 'basket_size'
						hue = 'power_user'
						fill = True,
						kind = 'kde')
plt.show()
```

Confounding = variable might cause the effect rather than treatment.

```python
# Example
# Randomly assign half
block_1 = productivity_subjects.sample(frac=0.5, random_state=42, replace=False)

# Set the block column
block_1['block'] = 1

# Create second assignment and label
block_2 = productivity_subjects.drop(block_1.index)
block_2['block'] = 2

# Concatenate and print
productivity_combined = pd.concat([block_1, block_2], axis=0)
print(productivity_combined['block'].value_counts())
```

### Stratified Randomization

Splitting based on covariate first then randomization

```python
strata_1 = ecom[ecom['power_user'] == 1]
strata_1['Block'] = 1
strata_1_g1 = strata_1.sample(frac=0.5, replace = False)
strata_1_g1['T_C'] = 'T'
strata_1_g2 = strata_1.drop(strata_1_g1.index)
strata_1_g2['T_C'] = 'C'

strata_2 = ecom.drop(strata_1.index)
strata_2['Block'] = 1
strata_2_g1 = strata_2.sample(frac=0.5, replace = False)
strata_2_g1['T_C'] = 'T'
strata_2_g2 = strata_2.drop(strata_2_g1.index)
strata_2_g2['T_C'] = 'C'

ecom_stratified = pd.concat([strata_1_g1, strata_1_g2, strata_2_g1, strata_2_g2])
ecom_stratified.groupby(['Block','T_C','power_user']).size()
```

```python
# Example
# Create the first block
strata_1 = wealth_data[wealth_data['high_wealth'] == 1]
strata_1['Block'] = 1

# Create two groups assigning to Treatment or Control
strata_1_g1 = strata_1.sample(100, replace=False)
strata_1_g1['T_C'] = 'T'
strata_1_g2 = strata_1.drop(strata_1_g1.index)
strata_1_g2['T_C'] = 'C'

# Create the second block and assign groups
strata_2 = wealth_data[wealth_data['high_wealth'] == 0]
strata_2['Block'] = 2

strata_2_g1 = strata_2.sample(900, replace=False)
strata_2_g1['T_C'] = 'T'
strata_2_g2 = strata_2.drop(strata_2_g1.index)
strata_2_g2['T_C'] = 'C'

# Concatenate the grouping work
wealth_data_stratified = pd.concat([strata_1_g1, strata_1_g2, strata_2_g1, strata_2_g2])
print(wealth_data_stratified.groupby(['Block','T_C', 'high_wealth']).size())
```

## Normal Data

Required for parametric tests. Nonparametric tests: do not assume normal data.

![image.png](attachment:e86eedfe-b8a7-43b3-bf87-133ccc7d4813:image.png)

```python
sns.displot(data = salaries, x = 'salary', kind = 'kde')
```

### QQ Plot

Compare data to a particular distribution

```python
from statsmodels.graphics.gofplots import qqplot
from scipy.stats.distributions import norm
qqplot(salaries['salary'],
				line = 's',
				dist = norm)
plt.show()
```

For an ideal output, dots should hug the line. On the other hand, dots bow out at ends.

Shapiro-Wilk → good for smaller datasets

D’Agostino K^2 → uses kurtosis and skewness 

```python
# Shapiro-Wilk Test
from scipy.stats import shapiro
alpha = 0.05
stat, p = shapiro(salaries['salary'])
print(f"p: {round(p,4)} test stat: {round(stat,4)}")
```

If p > alpha → Fail to reject H0 → likely normal

```python
# Anderson-Darling test
from scipy.stats import anderson
result = anderson(x=salaries['salary'], dist = 'norm')
```

```python
# Example
# Run the Anderson-Darling test
result = anderson(chicken_data['weight'], dist="norm")

print(f"Test statistic: {round(result.statistic, 4)}")
print(f"Significance Levels: {result.significance_level}")
print(f"Critical Values: {result.critical_values}")
```

# Experimental Design Techniques

## Factorial Designs: Principles and Applications

Factorial design data example

- Factor 1 (Light_Condition) - two levels: Full Sunlight and Partial Shade
- Factor 2 (Fertilizer_Type) - two levels: Synthetic and Organic
- Numeric response/dependent/outcome variable: Growth_cm

```python
plant_growth = pd.pivot_table(plant_growth_data,
															values='Growth_cm',
															index='Light_Condition',
															columns='Fertilizer_Type',
															aggfunc='mean')
															
sns.heatmap(plant_growth,
						annot = True,
						cmap = 'coolwarm',
						fmt = 'g')
plt.show()
```

Interaction: How the effect of one factor varies with the level of another factor. 

Significant interaction → factors do not work independently

```python
# Example
# Create a pivot table for marketing campaign data
marketing_pivot = marketing_data.pivot_table(
  values='Conversions', 
  index='Messaging_Style', 
  columns='Time_of_Day', 
  aggfunc='mean')

# View the pivoted results
print(marketing_pivot)
```

## Randomized Block Design: Controlling Variance

### Implementing Randomized Block Design

- Use .groupby() to shuffle within blocks

```python
blocks = athletes.groupby('Initial_Fitness_Level').apply(
		lambda x: x.sample(frac=1)
)
blocks = blocks.reset_index(drop=True)

blocks['Treatment'] = np.random.choice(
	['Cardio','Strength Training', 'Mixed'],
	size=len(blocks))
	
sns.boxplot(x='Initial_Fitness_Level', y='Muscle_Gain_kg', hue='Treatment', data=blocks)
plt.show()
```

```python
# Example
# Randomly assign workers to blocks
prod_df = productivity.groupby('block').apply(
  lambda x: x.sample(frac=1)
)

# Reset the index
prod_df = prod_df.reset_index(drop=True)

# Assign treatment randomly
prod_df['Treatment'] = np.random.choice(
  ['Bonus', 'Profit Sharing', 'Work from Home'],
  size=len(prod_df)
)

# Make a plot showing how positivity_score varies within blocks
sns.boxplot(x='block', 
            y='productivity_score', 
            hue='Treatment', 
            data=prod_df)

plt.show()

# Perform the within blocks ANOVA, first grouping by block
within_block_anova = prod_df.groupby('block').apply(
  # Set function
  lambda x: f_oneway(
    # Filter Treatment values based on outcome
    x[x['Treatment'] == 'Bonus']['productivity_score'], 
    x[x['Treatment'] == 'Profit Sharing']['productivity_score'],
    x[x['Treatment'] == 'Work from Home']['productivity_score'])
)
print(within_block_anova)
```

## Covariate Adjustment in Experimental Design

Covariates: Potentially affect experiment results but are not primary focus. Importance in reducing confounding. Impact on precision and validity of results.

```python
exp_plant_data = plant_data_growth[['Plant_ID', 'Fertilizer_Type','Growth_cm']]

merged_plant_data = pd.merge(exp_plant_data, covariate_data, on='Plant_ID')

from statsmodels.formula.api import ols
model = ols('Growth_cm ~ Fertilizer_Type + Watering_Days_Per_Week',
						data=merged_plant_data).fit()
```

```python
# Example
# Join experimental and covariate data
merged_chick_data = pd.merge(exp_chick_data, 
                             cov_chick_data, on='Chick')

# Perform ANCOVA with Diet and Time as predictors
model = ols('weight ~ Diet + Time', data=merged_chick_data).fit()

# Print a summary of the model
print(model.summary())

# Visualize Diet effects with Time adjustment
sns.lmplot(x='Time', y='weight', 
         hue='Diet', 
         data=merged_chick_data)
plt.show()
```

# Analyzing Experimental Data: Statistical Tests and Power

## Choosing the Right Statistical Test

### Independent samples t-test

Comparing means of two groups.

Assumptions: normal distribution, equal variances

```python
from scipy.stats import ttest_ind
group1 = athletic_perf[athletic_pref['Training_Program'] == 'HIIT']['Performance_Inc']
group2 = athletic_perf[athletic_perf['Training_Program'] == 'Endurance']['Performance_Inc']

t_stat, p_val = ttest_ind(group1, group2)
print(f"T-statistic:{t_stat}, p-value: {p_val}")
```

As a result, p_val > alpha then insufficient evidence of a difference in means

```python
# Example
# Separate the annual returns by strategy type
quantitative_returns = investment_returns[investment_returns['Strategy_Type'] == 'Quantitative']['Annual_Return']
fundamental_returns = investment_returns[investment_returns['Strategy_Type'] == 'Fundamental']['Annual_Return']

# Perform the independent samples t-test between the two groups
t_stat, p_val = ttest_ind(quantitative_returns, fundamental_returns)
print(print(f"T-statistic:{t_stat}, p-value: {p_val}"))
# T-statistic:7.784788496693728, p-value: 2.0567003424807146e-14

# Since p_val < alpha (0.1) then p-value is much smaller than alpha, suggesting a significant difference in return between two strategies.
```

### One-way ANOVA

Comparing means across multiple (>2) groups

Assumption: Equal variances among groups

```python
from scipy.stats import f_oneway
program_types = ['HIIT', 'Endurance', 'Strength']
groups = [athletic_perf_data[athletic_perf_data['Training_Program'] == program]
['Performance_Increase'] for program in program_types]
f_stat, p_val = f_oneway(*groups)
print(f"F-statistic: {f_stat}, P-value: {p_val}")
```

As a result, if p_val > alpha then insufficient evidence of a difference in means.

```python
# Example
catalyst_types = ['Palladium', 'Platinum', 'Nickel']

# Collect reaction times for each catalyst into a list
groups = [chemical_reactions[chemical_reactions['Catalyst'] == catalyst]['Reaction_Time'] for catalyst in catalyst_types]

# Perform the one-way ANOVA across the three groups
f_stat, p_val = f_oneway(*groups)
print(f"F-statistic: {f_stat}, P-value: {p_val}")

# The p_val is substantially smaller than the alpha value, indicating a important difference in reaction times across the catalysts.
```

### Chi-square Test of Association

Testing relationships between categorical variables

No assumptions about distributions

```python
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(athletic_perf['Training_Program'],
																athletic_perf['Diet_Type'])
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2-statistic: {chi2_stat}, P-value: {p_val}")
```

As a result, if p_val > alpha then insufficient evidence of an association

```python
# Example
# Create a contingency table
contingency_table = pd.crosstab(
  hr_wellness['Department'], 
  hr_wellness['Wellness_Program_Status']
)

# Perform the chi-square test of association
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2-statistic: {chi2_stat}, P-value: {p_val}")
# According to the p value and alpha values, there is not significant association
# between department and enrollment in the wellness program. Since p_val > alpha
```

## Post-Hoc Analysis Following ANOVA

```python
from scipy.stats import f_oneway
campaign_types = ['Seasonal Discount','New Arrival', 'Loyalty Reward']
groups = [ad_campaigns[ad_campaigns['Ad_Campaign'] == campaign]
['Click_Through_Rate'] for campaign in campaign_types]

f_stat, p_val = f_oneway(*groups)
print(p_val) -> 4.48412e-134
# Significant differences in means
```

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_reuslts = pairwise_tukeyhsd(ad_campaigns['Click_Through_Rate'],
																	ad_campaigns['Ad_Campaign'],
																	alpha = 0.05)
```

![image.png](attachment:18026ede-eb5d-4d81-9de3-29a9410838c7:image.png)

```python
# Bonferroni Correction set-up
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests

p_values = []
comparisons = [('Seasonal Discount','New Arrival'),
							 ('Seasonal Discount','Loyalty Reward'),
							 ('New Arrival','Loyalty Reward')]
							 
for comp in comparisons:
		group1 = ad_campaigns[ad_campaigns['Ad Campaign'] == comp[0]]['Click_Through_Rate']
		group2 = ad_campaigns[ad_campaigns['Ad Campaign'] == comp[1]]['Click_Through_Rate']
		t_stat, p_val = ttest_ind(group1, group2)
		p.values.append(p_val)
		
p_adjusted = multipletests(p_values, alpha=0.05, method='bonferroni')
print(f"Adjusted P-values: {p_adjusted[1]}")
```

## P-Values, Alpha and Errors

P-values: Probability of observing our data if the null hypothesis was true

alpha: Threshold at which we consider our results statistically significant

If p-value ≤ alpha then reject null hypothesis in favor of alternative hypothesis.

```python
from scipy.stats import ttest_ind
organic_yield = crop_yields[crop_yields['Fertilizer_Type'] == 'Organic']['Crop_Yield']
synthetic_yield = crop_yields[crop_yields['Fertilizer_Type'] == 'Synthetic']['Crop_Yield']
t_stat, p_val = ttest_ind(organic_yield, synthetic_yield)
print(p_val)
```

![image.png](attachment:321aac2e-3d71-483f-978f-9993aa774d50:image.png)

### More on Alpha

- Common values are 0.05, 0.01 and 0.1, which means 5%, 1% and 10% probability of making a Type 1 error
- Choosing and alpha, based on the context of the study, balancing a tolerance for a Type 1 error.
- Conventions: 
- 0.05 (5%): Most common, used as a convention
- 0.01 (1%): More stringest testing, where cost of Type 1 error is high
- 0.1 (10%): Sometimes in preliminary studies, where higher tolerance for Type 1 error

```python
# Example
# Calculate mean Durability_Score for each Toy_Type
mean_durability = toy_durability.pivot_table(
  values='Durability_Score', index='Toy_Type', aggfunc="mean")
print(mean_durability)

# Perform t-test
educational_durability = toy_durability[toy_durability['Toy_Type'] == 'Educational']['Durability_Score']
recreational_durability = toy_durability[toy_durability['Toy_Type'] == 'Recreational']['Durability_Score']
t_stat, p_val = ttest_ind(educational_durability, recreational_durability)

print(p_val)

# Visualize the distribution of Durability_Score for each Toy_Type
sns.displot(data=toy_durability, x="Durability_Score", 
         hue="Toy_Type", kind="kde")
plt.title('Durability Score Distribution by Toy Type')
plt.xlabel('Durability Score')
plt.ylabel('Density')
plt.show()
```

## Power Analysis: Sample and Effect Size

Effect size quantifies the difference between two groups.

Cohen’s d is standard measure for effect size.

Power: The probability of correctly rejecting a false null hypothesis (1- ß). It ranges between 0 and 1 (certainty in ability to detect a true effect).

To calculate power:

- Assume effect_size = 1 from historical data

```python
from statsmodels.stats.power import TTestIndPower
power_analysis = TTestIndPower()
power = power_analysis.solve_power(effect_size=1, nobs1=30, alpha=0.05)
```

```python
# Cohen's d formulation
def cohens_d(group1, group2):
		diff = group1.mean() - group2.mean()
		n1, n2 = len(group1), len(group2)
		var1, var2 = group1.var(), group2.var()
		pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 -2))
		d = diff / pooled_std
		return d
```

```python
power_analysis = TTestIndPower()
required_n = power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.99, ratio=1)
```

# Advanced Insights from Experimental Complexity

## Synthesizing Insights from Complex Experiments

## Addressing Complexities in Experimental Data

Heteroscedasticity: Changing variability of a variable across the range of another variable

```python
# Check for heteroscedasticity with a residual plot
sns.residplot(x='NutrientRetention', y='ShelfLife', 
         data=food_preservation, lowess=True)
plt.title('Residual Plot of Shelf Life and Nutrient Retention')
plt.xlabel('Nutrient Retention (%)')
plt.ylabel('Residuals')
plt.show()
```

```python
# Visualize the original ShelfLife distribution
sns.displot(food_preservation['ShelfLife'])
plt.title('Original Shelf Life Distribution')
plt.show()

# Create a Box-Cox transformation
ShelfLifeTransformed, _ = boxcox(food_preservation['ShelfLife'])

# Visualize the transformed ShelfLife distribution
plt.clf()
sns.displot(ShelfLifeTransformed)
plt.title('Transformed Shelf Life Distribution')
plt.show()
```

## Applying Nonparametric Tests in Experimental Analysis

If parametric test assumptions not met, then we use nonparametric tests. Data on ordinal scale or non-normal. Robust to outliers and non-linear data.

Nonparametric methods for data that does not fit traditional assumptions. 

- Mann-Whitney U Test: Compare two independent groups.
- Kruskal-Wallis Test: Compare more than two groups

```python
# Ensure the 'NutrientRetention' column is numeric
food_preservation['NutrientRetention'] = pd.to_numeric(food_preservation['NutrientRetention'], errors='coerce')

# Separate nutrient retention for Freezing and Canning methods
freezing = food_preservation[food_preservation['PreservationMethod'] == 'Freezing']['NutrientRetention']
canning = food_preservation[food_preservation['PreservationMethod'] == 'Canning']['NutrientRetention']

# Perform Mann Whitney U test
u_stat, p_val = mannwhitneyu(freezing, canning)
```