# 🏠 Exploring Airbnb Market Trends

## Project Overview
Market analysis of Airbnb listings in New York to investigate the short-term rental market and provide strategic insights on private rooms for a real estate start-up.

## 📊 Project Objectives
- Analyze Airbnb listing data from multiple sources
- Investigate private room market segment
- Determine pricing trends and review patterns
- Provide data-driven insights for real estate investment decisions

## 🔍 Key Analysis Questions
1. What are the dates of the earliest and most recent reviews?
2. How many of the listings are private rooms?
3. What is the average listing price?
4. How can we combine these insights into a comprehensive summary?

## 📈 Key Findings
- **Review Timeline**: Identified the full span of review activity from earliest to most recent dates
- **Private Room Market**: Quantified the number of private room listings in the dataset
- **Pricing Analysis**: Calculated average listing price across all properties
- **Market Summary**: Consolidated key metrics into single comprehensive overview

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and analysis across multiple file formats
- **File Processing**: CSV, Excel (.xlsx), and TSV file handling

## 📁 Project Structure
```
exploring-airbnb-market-trends/
├── README.md              # This file
├── data/                  # Multiple Airbnb data files
│   ├── airbnb_price.csv          # Pricing information
│   ├── airbnb_room_type.xlsx     # Room type classifications  
│   └── airbnb_last_review.tsv    # Review date information
├── notebooks/             # Jupyter notebooks with analysis
├── images/                # Market visualizations
└── src/                   # Python analysis scripts
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas openpyxl jupyter
```

### Data Files
The analysis uses three complementary datasets:
- **airbnb_price.csv**: Listing pricing information
- **airbnb_room_type.xlsx**: Room type categorizations
- **airbnb_last_review.tsv**: Review timeline data

## 💡 Key Analysis Steps
1. **Multi-format Data Loading**: Import data from CSV, Excel, and TSV sources
2. **Review Timeline Analysis**: Extract earliest and most recent review dates
3. **Room Type Filtering**: Count private room listings specifically
4. **Pricing Analysis**: Calculate average listing price with proper rounding
5. **Data Consolidation**: Combine insights into unified summary DataFrame

## 🎯 Skills Demonstrated
- **Multi-format Data Handling**: Working with CSV, Excel, and TSV files
- **Date Analysis**: Finding minimum and maximum date values
- **Categorical Filtering**: Counting specific room type categories
- **Statistical Analysis**: Average price calculation with rounding
- **Data Integration**: Combining multiple metrics into structured output
- **Real Estate Analytics**: Market-focused data interpretation

## 📊 Deliverables
**Final Output**: `review_dates` DataFrame containing:
- `first_reviewed`: Earliest review date in dataset
- `last_reviewed`: Most recent review date in dataset  
- `nb_private_rooms`: Total count of private room listings
- `avg_price`: Average listing price (rounded to 2 decimal places)

## 🏙️ Business Impact
**For Real Estate Start-up:**
- **Market Timing**: Understanding review activity timeline for market maturity assessment
- **Segment Focus**: Quantified private room opportunity for investment targeting
- **Pricing Strategy**: Baseline pricing data for competitive positioning
- **Investment Decision**: Consolidated metrics for strategic planning

## 📈 Market Insights
This analysis provides essential baseline metrics for:
- **Private Room Investment**: Understanding market size and opportunity
- **Pricing Strategy**: Competitive pricing based on market averages
- **Market Maturity**: Review timeline indicating platform adoption
- **Portfolio Planning**: Data-driven foundation for property acquisition

## 🔗 Related Projects
- [Customer Analytics Data Modeling](../customer-analytics-data-modeling/)
- [Analyzing Crime in LA](../analyzing-crime-in-la/)

## 📄 Data Sources
Airbnb listing data from various sources including pricing, room types, and review information for New York market analysis.

---

**Project Status**: ✅ Completed  
**Last Updated**: [Date]  
**Business Focus**: Private room market analysis for real estate investment