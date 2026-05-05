# 🧠 Analyzing Students' Mental Health

## Project Overview
SQL-based analysis exploring how the length of stay impacts mental health diagnostic scores for international students. The analysis aggregates standardized test results across different stay durations to surface trends relevant to student well-being research.

## 📊 Project Objectives
- Investigate the `students` dataset to understand schema and key variables
- Filter the dataset to international students with a recorded length of stay
- Aggregate diagnostic scores (PHQ-9, SCS, ASISS) by length of stay
- Produce a clean summary table sorted by length of stay (descending)

## 🔍 Key Analysis Components

### Data Investigation
- **Dataset Exploration**: Initial inspection of the `students` table
- **Population Filter**: Restrict to international students (`inter_dom = 'Inter'`)
- **Quality Filter**: Exclude rows with missing length of stay (`stay IS NOT NULL`)

### Aggregation Logic
- **Group key**: `stay` (length of stay in years)
- **Counts**: Number of international students per stay value
- **Averages**: Mean diagnostic scores per stay value, rounded to 2 decimals
  - `todep` → PHQ-9 (depression)
  - `tosc`  → SCS (social connectedness)
  - `toas`  → ASISS (acculturative stress)

## 📈 Key Deliverable
**Primary Output**: A 9-row, 5-column result set with columns:

| Column        | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `stay`        | Length of stay (years)                                       |
| `count_int`   | Number of international students for each length of stay    |
| `average_phq` | Average PHQ-9 (`todep`) score, rounded to 2 decimals        |
| `average_scs` | Average SCS (`tosc`) score, rounded to 2 decimals           |
| `average_as`  | Average ASISS (`toas`) score, rounded to 2 decimals         |

Sorted by `stay` in descending order.

## 🛠️ Technologies Used
- **PostgreSQL**: Query execution and aggregation
- **SQL**: `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `COUNT`, `AVG`, `ROUND`

## 📁 Project Structure
```
student-mental-health-analysis/
├── README.md                              # This file
└── query/
    └── student-mental-health-analysis.sql # Final analysis query
```

## 🚀 Getting Started

### Prerequisites
- A running PostgreSQL instance with the `students` table loaded
- SQL client (psql, DBeaver, DataGrip, or DataCamp's in-browser editor)

### Running the Query
```sql
\i query/student-mental-health-analysis.sql
```

## 💡 Key Analysis Steps
1. **Inspect the data**: `SELECT * FROM students;` to understand columns and types
2. **Filter the population**:
   - Keep only international students (`inter_dom = 'Inter'`)
   - Drop rows where `stay` is `NULL`
3. **Aggregate by stay length**:
   - Count rows per group as `count_int`
   - Compute rounded averages for `todep`, `tosc`, `toas`
4. **Sort**: Order results by `stay` descending
5. **Validate**: Confirm output has exactly nine rows and five aliased columns

## 🎯 Skills Demonstrated
- **SQL Aggregation**: `GROUP BY` with multiple aggregate functions
- **Filtering**: Combining `WHERE` clauses with `IS NOT NULL` and equality checks
- **Output Shaping**: Aliasing columns and rounding numeric output
- **Result Ordering**: Sorting on the grouping key
- **Healthcare/Behavioral Data Analysis**: Working with standardized diagnostic instruments

## 🧠 Diagnostic Scales Reference
- **PHQ-9 (`todep`)**: Patient Health Questionnaire — measures depression severity
- **SCS (`tosc`)**: Social Connectedness Scale — measures sense of belonging
- **ASISS (`toas`)**: Acculturative Stress Scale for International Students

## 🔬 Methodology
1. **Schema discovery**: Survey columns relevant to mental health and demographics
2. **Population definition**: Limit to international students only
3. **Aggregation**: Compute per-`stay` summary statistics
4. **Presentation**: Round and alias columns for downstream readability
5. **Sorting**: Apply descending sort for natural reporting order

## 🔗 Related Work
- [Hypothesis Testing with Soccer Matches](../hypothesis-testing-soccer-matches/)
- [Course Notes: Introduction to SQL](../../course-notes/introduction-to-sql/)
- [Course Notes: Intermediate SQL](../../course-notes/intermediate-sql/)

## 📄 Data Source
The `students` table contains demographic information and standardized mental health diagnostic scores for domestic and international students.

---

**Project Status**: ✅ Completed
**Research Focus**: Cross-sectional SQL analysis of international student mental health by length of stay
