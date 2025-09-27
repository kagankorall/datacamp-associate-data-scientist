# ⚽ Hypothesis Testing with Men's and Women's Soccer Matches

## Project Overview
Statistical hypothesis testing project to compare goal-scoring patterns between men's and women's international soccer matches using FIFA World Cup data and rigorous statistical methods.

## 📊 Project Objectives
- Compare mean number of goals scored in men's vs women's international soccer matches
- Apply appropriate hypothesis testing methodology
- Use FIFA World Cup official match data since 2002
- Determine statistical significance with 10% significance level

## 🔍 Hypothesis Testing Framework

### Research Question
**Is there a significant difference in the mean number of goals scored between men's and women's international soccer matches?**

### Hypotheses
- **Null Hypothesis (H₀)**: The mean number of goals scored in women's international soccer matches is the same as men's
- **Alternative Hypothesis (H₁)**: The mean number of goals scored differs between men's and women's matches

### Test Parameters
- **Significance Level (α)**: 10% (0.10)
- **Data Period**: FIFA World Cup matches since 2002-01-01
- **Assumption**: Each match is fully independent (team form ignored)

## 📈 Key Analysis Components
- **Data Filtering**: Official FIFA World Cup matches from 2002 onwards
- **Statistical Test**: Appropriate hypothesis test for comparing means
- **P-value Calculation**: Quantify statistical evidence
- **Decision Making**: Reject or fail to reject null hypothesis based on significance level

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and filtering
- **scipy**: Statistical hypothesis testing
- **Statistical analysis**: Two-sample hypothesis testing

## 📁 Project Structure
```
hypothesis-testing-soccer-matches/
├── README.md              # This file
├── data/                  # FIFA World Cup match data
├── notebooks/             # Jupyter notebooks with analysis
├── images/                # Statistical visualizations
└── src/                   # Python analysis scripts
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scipy numpy jupyter matplotlib
```

### Dataset
FIFA World Cup official match data containing goal statistics for men's and women's international matches since 2002.

## 💡 Key Analysis Steps
1. **Data Preparation**: Filter FIFA World Cup matches since 2002-01-01
2. **Data Segmentation**: Separate men's and women's match data
3. **Goal Calculation**: Calculate total goals per match for each group
4. **Statistical Testing**: Apply appropriate two-sample hypothesis test
5. **P-value Interpretation**: Compare p-value to 10% significance level
6. **Decision Making**: Determine whether to reject or fail to reject H₀

## 🎯 Skills Demonstrated
- **Hypothesis Testing**: Proper statistical test selection and execution
- **Data Filtering**: Time-based and categorical data filtering
- **Statistical Interpretation**: P-value analysis and decision making
- **Sports Analytics**: Soccer match data analysis
- **Research Methodology**: Independent sample assumptions and study design

## 📊 Deliverables
**Primary Output**: `result_dict` containing:
- `"p_val"`: Calculated p-value from hypothesis test
- `"result"`: Decision outcome (`"reject"` or `"fail to reject"`)

**Format**:
```python
result_dict = {"p_val": p_val, "result": result}
```

## ⚽ Sports Analytics Applications
This analysis provides insights for:
- **Game Strategy**: Understanding scoring patterns across men's and women's soccer
- **Tournament Planning**: Evidence-based expectations for goal scoring
- **Broadcasting**: Statistical context for match commentary
- **Performance Analysis**: Baseline metrics for team evaluation

## 🔬 Statistical Methodology
1. **Sample Independence**: Each match treated as independent observation
2. **Appropriate Test Selection**: Choose suitable test for comparing means
3. **Significance Testing**: 10% significance level application
4. **Result Interpretation**: Clear decision based on statistical evidence

## 📈 Expected Outcomes
- **Statistical Evidence**: Quantified difference (or lack thereof) in goal scoring
- **Decision Framework**: Clear methodology for hypothesis acceptance/rejection
- **Sports Insights**: Data-driven understanding of soccer scoring patterns
- **Methodological Foundation**: Replicable approach for similar sports comparisons

## 🔗 Related Projects
- [Modeling Car Insurance Claim Outcomes](../modeling-car-insurance-claim-outcomes/)
- [Clustering Antarctic Penguin Species](../clustering-antarctic-penguin-species/)

## 📄 Data Sources
Official FIFA World Cup match data including goal statistics, match dates, and tournament information for comprehensive hypothesis testing.

---

**Project Status**: ✅ Completed  
**Last Updated**: [Date]  
**Statistical Focus**: Two-sample hypothesis testing for sports analytics