# 🚗 Modeling Car Insurance Claim Outcomes

## Project Overview
Predictive modeling project to identify the most important features for predicting car insurance claim outcomes and build accurate classification models for risk assessment.

## 📊 Project Objectives
- Identify the single best predictor of insurance claim outcomes
- Evaluate feature importance through accuracy scoring
- Build predictive models for claim outcome classification
- Provide insights for insurance risk assessment and pricing

## 🔍 Key Analysis Question
**Primary Goal**: Identify the single feature (excluding `"id"` column) that is the best predictor of whether a customer will put in a claim (the `"outcome"` column).

## 📈 Key Findings
- **Best Predictor Feature**: Identified the single most predictive feature for claim outcomes
- **Accuracy Assessment**: Quantified predictive power through accuracy scoring
- **Risk Factors**: Determined which customer/policy characteristics best predict claims

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models and accuracy evaluation
- **Classification modeling**: Feature evaluation and prediction

## 📁 Project Structure
```
modeling-car-insurance-claim-outcomes/
├── README.md              # This file
├── data/                  # Insurance claim dataset
├── notebooks/             # Jupyter notebooks with modeling
├── images/                # Feature importance visualizations
└── src/                   # Python modeling scripts
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn jupyter matplotlib seaborn
```

### Dataset
Car insurance dataset containing customer and policy features with binary claim outcomes.

## 💡 Key Analysis Steps
1. **Data Exploration**: Analyze all features excluding the `"id"` column
2. **Feature Evaluation**: Test each feature individually as predictor for `"outcome"`
3. **Accuracy Calculation**: Measure prediction accuracy for each feature
4. **Best Feature Selection**: Identify feature with highest accuracy score
5. **Result Documentation**: Store findings in structured DataFrame format

## 🎯 Skills Demonstrated
- **Feature Selection**: Systematic evaluation of individual predictors
- **Classification Modeling**: Building predictive models for binary outcomes
- **Model Evaluation**: Accuracy assessment and comparison
- **Insurance Analytics**: Understanding risk factors in insurance context
- **Data-Driven Decision Making**: Objective feature importance ranking

## 📊 Deliverables
**Primary Output**: `best_feature_df` DataFrame containing:
- `"best_feature"`: Name of the feature with highest predictive accuracy
- `"best_accuracy"`: Corresponding accuracy score for the best feature

## 🏢 Business Impact
**For Insurance Industry:**
- **Risk Assessment**: Identify key factors predicting claim likelihood
- **Underwriting**: Focus on most predictive features for policy pricing
- **Data Strategy**: Prioritize data collection for high-impact variables
- **Model Development**: Foundation for more complex predictive models

## 📈 Insurance Applications
This analysis enables:
- **Premium Pricing**: Risk-based pricing using best predictive features
- **Customer Segmentation**: Group customers by claim risk probability
- **Policy Optimization**: Adjust coverage based on risk factors
- **Fraud Detection**: Identify unusual patterns in high-risk features

## 🔍 Methodology
1. **Feature Isolation**: Evaluate each feature independently
2. **Model Training**: Train classification models using single features
3. **Performance Measurement**: Calculate accuracy for each feature-based model
4. **Ranking**: Identify feature with maximum accuracy
5. **Documentation**: Structure results for business consumption

## 🔗 Related Projects
- [Customer Analytics Data Modeling](../customer-analytics-data-modeling/)
- [Hypothesis Testing with Soccer Matches](../hypothesis-testing-soccer-matches/)

## 📄 Data Sources
Car insurance dataset containing customer demographics, policy details, and historical claim outcomes for predictive modeling.

---

**Project Status**: ✅ Completed  
**Last Updated**: [Date]  
**Business Value**: Risk prediction and feature importance analysis