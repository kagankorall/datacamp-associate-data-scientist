# 🏆 Nobel Prize Coding

## Project Overview
Analysis of Nobel Prize winners and patterns in awards across different categories and time periods using Python for data manipulation and visualization.

## 📊 Project Objectives
- Analyze historical trends in Nobel Prize awards
- Identify patterns in prize categories over time
- Examine demographic distributions of Nobel laureates
- Visualize key insights about Nobel Prize data

## 🔍 Key Questions Explored
1. What is the most commonly awarded gender and birth country?
2. Which decade had the highest ratio of US-born Nobel Prize winners to total winners?
3. Which decade and Nobel Prize category combination had the highest proportion of female laureates?
4. Who was the first woman to receive a Nobel Prize, and in what category?
5. Which individuals or organizations have won more than one Nobel Prize throughout the years?

## 📈 Key Findings
- **Most Common Demographics**: Analysis revealed the predominant gender and birth country of Nobel laureates
- **US Winners Peak Decade**: Identified the decade with highest ratio of US-born laureates to total winners
- **Female Representation Peak**: Found the specific decade and category combination with highest proportion of female laureates  
- **First Female Laureate**: Discovered the pioneering woman who broke barriers in Nobel Prize history
- **Multiple Winners**: Identified individuals and organizations who achieved the rare feat of winning multiple Nobel Prizes

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **matplotlib & seaborn**: Data visualization
- **numpy**: Numerical computations

## 📁 Project Structure
```
nobel-prize-coding/
├── README.md              # This file
├── data/                  # Nobel Prize dataset
├── notebooks/             # Jupyter notebooks with analysis
├── images/                # Generated plots and visualizations
└── src/                   # Python scripts (if any)
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas matplotlib seaborn numpy jupyter
```

### Running the Analysis
1. Clone the repository
2. Navigate to the project directory
3. Open the Jupyter notebook
4. Run all cells to reproduce the analysis

## 📊 Sample Visualizations
**[Add screenshots of your key visualizations here]**

## 💡 Key Analysis Steps
1. **Demographic Analysis**: Determining most common gender and birth country using mode calculations
2. **Temporal Analysis**: Creating decade variables and calculating US-born winner ratios by decade
3. **Gender Representation Study**: Analyzing female laureate proportions across decades and categories
4. **Historical Research**: Identifying the first female Nobel Prize winner through chronological sorting
5. **Multiple Winner Detection**: Finding repeat laureates using groupby operations and filtering

## 🎯 Skills Demonstrated
- **Data manipulation**: Using pandas for filtering, grouping, and aggregating data
- **Time series analysis**: Creating decade variables and temporal comparisons
- **Statistical analysis**: Calculating ratios, proportions, and identifying trends
- **Data exploration**: Finding patterns in gender, geography, and category distributions
- **Historical analysis**: Chronological data sorting and milestone identification

## 📝 Methodology
1. **Data Loading**: Import Nobel Prize dataset using pandas
2. **Demographic Analysis**: 
   - Calculate mode values for gender and birth country
   - Identify most common laureate characteristics
3. **Decade Creation**: Extract and create decade variables from year data
4. **Ratio Calculations**: 
   - Filter US-born winners by decade
   - Calculate ratios of US winners to total winners
   - Identify peak decade for US representation
5. **Gender Proportion Analysis**:
   - Group data by decade and category
   - Calculate female laureate proportions
   - Find maximum female representation combination
6. **Historical Firsts**: Sort chronologically to find first female laureate
7. **Multiple Winner Detection**: Use groupby filtering to identify repeat winners

## 🔧 Technical Implementation
```python
# Key code patterns used:
# Mode calculation for demographics
top_gender = nobel_df['sex'].mode()[0]

# Decade extraction and ratio analysis  
nobel_df['decade'] = (nobel_df['year'] // 10) * 10
ratio_per_decade = us_winners_per_decade / total_winners_per_decade

# Groupby operations for proportion analysis
female_proportion = nobel_df[nobel_df['sex'] == 'Female'].groupby(['decade', 'category']).size()

# Filtering for repeat winners
repeat_winners = nobel_df.groupby('full_name').filter(lambda x: len(x) > 1)
```

## 🔗 Related Projects
- [Analyzing Crime in LA](../analyzing-crime-in-la/)
- [Hypothesis Testing with Soccer Matches](../hypothesis-testing-soccer-matches/)

## 📄 Data Sources
Nobel Prize data from official Nobel Prize organization records.

---

**Project Status**: ✅ Completed  
**Last Updated**: [Date]  
**Analysis Period**: Historical Nobel Prize data