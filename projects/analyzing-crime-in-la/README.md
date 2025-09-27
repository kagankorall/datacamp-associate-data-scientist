# 🚨 Analyzing Crime in LA

## Project Overview
Comprehensive analysis of crime patterns and trends in Los Angeles using real crime data to identify temporal patterns, geographical hotspots, and victim demographics.

## 📊 Project Objectives
- Identify peak crime hours and temporal patterns
- Analyze geographical distribution of night crimes
- Examine victim age demographics across different crime types
- Understand crime patterns for public safety insights

## 🔍 Key Questions Explored
1. Which hour has the highest frequency of crimes?
2. Which area has the largest frequency of night crimes (10pm-3:59am)?
3. What is the distribution of crimes across different victim age groups?

## 📈 Key Findings
- **Peak Crime Hour**: Identified the specific hour with highest crime frequency
- **Night Crime Hotspot**: Discovered the area with most crimes during night hours (10pm-3:59am)
- **Victim Age Analysis**: Analyzed crime distribution across age groups (0-17, 18-25, 26-34, 35-44, 45-54, 55-64, 65+)

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations and data categorization
- **matplotlib & seaborn**: Data visualization
- **Data processing**: String manipulation and categorical binning

## 📁 Project Structure
```
analyzing-crime-in-la/
├── README.md              # This file
├── data/                  # Crime dataset (crimes.csv)
├── notebooks/             # Jupyter notebooks with analysis
├── images/                # Generated plots and visualizations
└── src/                   # Python scripts (if any)
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Running the Analysis
1. Clone the repository
2. Navigate to the project directory
3. Ensure `crimes.csv` is in the data/ folder
4. Open the Jupyter notebook
5. Run all cells to reproduce the analysis

## 💡 Key Analysis Steps
1. **Time Extraction**: Extract hour information from TIME OCC string data
2. **Peak Hour Analysis**: Use mode calculation to find most frequent crime hour
3. **Night Crime Classification**: 
   - Define night hours (10pm-3:59am) using boolean conditions
   - Categorize crimes into "night" vs "morning" periods
   - Identify geographical areas with highest night crime frequency
4. **Age Group Analysis**:
   - Create age bins (0-17, 18-25, 26-34, 35-44, 45-54, 55-64, 65+)
   - Use pandas cut function to categorize victim ages
   - Generate frequency distribution across age groups

## 🎯 Skills Demonstrated
- **String manipulation**: Extracting hour data from time strings
- **Categorical analysis**: Creating and working with categorical variables
- **Boolean indexing**: Complex filtering conditions for night crime analysis
- **Data binning**: Using pandas cut for age group categorization
- **Geographical analysis**: Area-based crime frequency analysis
- **Temporal analysis**: Time-based pattern identification

## 📝 Methodology
1. **Data Loading**: Import LA crime dataset with proper data types
2. **Time Processing**: 
   ```python
   crimes["HOUR_OCC"] = crimes["TIME OCC"].str[:2].astype(int)
   peak_crime_hour = crimes["HOUR_OCC"].mode()[0]
   ```
3. **Night Crime Analysis**:
   ```python
   night_hours = [(crimes["HOUR_OCC"] >= 22) | (crimes["HOUR_OCC"] <= 3)]
   crimes["DAY_OR_NIGHT"] = np.select(night_hours, ["night"], default="morning")
   ```
4. **Age Group Categorization**:
   ```python
   age_bins = [0,17,25,34,44,54,64,np.inf]
   crimes["AGE_LEVEL"] = pd.cut(crimes["Vict Age"], labels=age_labels, bins=age_bins)
   ```

## 🔧 Technical Implementation
**Key Data Transformations:**
- String slicing for hour extraction from time data
- Boolean logic for night/day categorization
- Pandas cut function for age group binning
- Value counts for frequency analysis across categories

## 📊 Expected Outputs
- **Peak Crime Hour**: Integer representing hour with most crimes
- **Night Crime Location**: String with area name having highest night crime frequency  
- **Victim Age Distribution**: Pandas Series with age group frequencies

## 🔗 Related Projects
- [Nobel Prize Coding](../nobel-prize-coding/)
- [Customer Analytics Data Modeling](../customer-analytics-data-modeling/)

## 📄 Data Sources
Los Angeles Police Department crime data containing incident details, timing, location, and victim information.

## 🏙️ Public Safety Insights
This analysis provides valuable insights for:
- **Law Enforcement**: Resource allocation during peak crime hours
- **Urban Planning**: Focus on high-crime areas for safety improvements
- **Community Safety**: Understanding vulnerable demographics and time periods

---

**Project Status**: ✅ Completed  
**Last Updated**: [Date]  
**Data Period**: LA Crime Records