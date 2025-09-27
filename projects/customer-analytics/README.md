# 👥 Customer Analytics: Preparing Data For Modelling

## Project Overview
Data preprocessing and optimization project for Training Data Ltd. to efficiently prepare customer data for machine learning models by optimizing data types and implementing strategic filtering.

## 📊 Project Objectives
- Optimize DataFrame memory usage through efficient data type conversions
- Implement proper categorical data handling (ordered vs unordered)
- Apply business logic filtering for target customer segments
- Prepare clean, model-ready dataset for downstream analytics

## 🔍 Business Requirements
**Data Type Optimization:**
1. Two-factor categories → Boolean (`bool`)
2. Integer-only columns → 32-bit integers (`int32`) 
3. Float columns → 16-bit floats (`float16`)
4. Nominal categorical data → `category` data type
5. Ordinal categorical data → Ordered categories with natural ordering

**Business Filtering:**
- Focus on experienced professionals (10+ years experience)
- Target enterprise companies (1000+ employees)
- Align with recruiter base specialization

## 📈 Key Transformations

### Data Type Optimizations
- **Boolean Conversion**: `relevant_experience`, `job_change`
- **Integer Optimization**: `student_id`, `training_hours` → int32
- **Float Optimization**: `city_development_index` → float16
- **Ordered Categories**: `enrolled_university`, `education_level`, `experience`, `company_size`, `last_new_job`
- **Standard Categories**: Remaining object columns

### Business Logic Filtering
- **Experience Filter**: ≥ 10 years of professional experience
- **Company Size Filter**: ≥ 1000 employees (enterprise focus)

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and type optimization
- **Memory optimization**: Efficient data type selection

## 📁 Project Structure
```
customer-analytics-data-modeling/
├── README.md              # This file
├── data/                  # Customer training dataset
├── notebooks/             # Jupyter notebooks with preprocessing
├── images/                # Memory usage comparisons
└── src/                   # Python preprocessing scripts
```

## 💡 Key Technical Implementation

### Ordered Category Mapping
```python
ordered_cats = {
    'enrolled_university': ['no_enrollment', 'Part time course', 'Full time course'],
    'education_level': ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'],
    'experience': ['<1'] + list(map(str, range(1, 21))) + ['>20'],
    'company_size': ['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'],
    'last_new_job': ['never', '1', '2', '3', '4', '>4']
}
```

### Boolean Mapping
```python
two_factor_cats = {
    'relevant_experience': {'No relevant experience': False, 'Has relevant experience': True},
    'job_change': {0.0: False, 1.0: True}
}
```

### Dynamic Type Conversion
```python
for col in ds_jobs_transformed:
    if col in ['relevant_experience', 'job_change']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].map(two_factor_cats[col])
    elif col in ['student_id', 'training_hours']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('int32')
    # ... additional type conversions
```

## 🎯 Skills Demonstrated
- **Memory Optimization**: Strategic data type selection for reduced memory footprint
- **Categorical Data Handling**: Proper distinction between ordinal and nominal categories
- **Data Type Conversion**: Efficient pandas data type transformations
- **Business Logic Implementation**: Filtering based on domain requirements
- **Exploratory Data Analysis**: Understanding data distributions for optimal preprocessing
- **Pipeline Development**: Systematic data transformation workflow

## 📊 Performance Improvements
- **Memory Usage Reduction**: Significant decrease through optimized data types
- **Processing Efficiency**: Improved performance for downstream ML models
- **Data Quality**: Clean, consistently typed dataset ready for modeling

## 🔧 Methodology
1. **Initial Data Exploration**: Analyze data types and unique values
2. **Category Identification**: Distinguish between ordered, nominal, and binary categories
3. **Mapping Creation**: Define transformation dictionaries for systematic conversion
4. **Type Optimization**: Apply memory-efficient data types column by column
5. **Business Filtering**: Apply experience and company size requirements
6. **Validation**: Compare memory usage before and after optimization

## 📈 Expected Outcomes
- **Optimized DataFrame**: `ds_jobs_transformed` with reduced memory usage
- **Filtered Dataset**: Focused on target demographic (experienced enterprise professionals)
- **Model-Ready Data**: Clean, properly typed data for machine learning pipelines
- **Performance Metrics**: Substantial memory usage reduction measurable via `.info()` and `.memory_usage()`

## 🏢 Business Impact
**For Training Data Ltd.:**
- **Resource Efficiency**: Reduced computational costs through memory optimization
- **Target Focus**: Data aligned with recruiter specialization in enterprise clients
- **Model Performance**: Clean, optimized data improves ML model training efficiency
- **Scalability**: Efficient data handling for larger datasets

## 🔗 Related Projects
- [Nobel Prize Coding](../nobel-prize-coding/)
- [Analyzing Crime in LA](../analyzing-crime-in-la/)

## 📄 Data Sources
Training Data Ltd. customer training dataset (`customer_train.csv`) containing job seeker profiles and employment history.

---

**Project Status**: ✅ Completed  
**Last Updated**: [Date]  
**Business Value**: Memory optimization + targeted customer segmentation