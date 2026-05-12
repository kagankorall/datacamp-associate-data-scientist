# 🎮 The Golden Era of Video Games

## Project Overview
SQL-based analysis exploring video game critic and user scores alongside sales data for the top 400 video games released since 1977. The project searches for a "golden age" of video games by identifying release years that critics and users rated highly, and surveys the business side of gaming through top-selling titles.

## 📊 Project Objectives
- Identify the ten best-selling video games by copies sold
- Surface the ten release years with the highest average critic score (minimum sample size of four games)
- Locate the "golden years" where critics and users broadly agreed games were highly rated
- Practice multi-table joins, aggregation, filtering, and set theory across two core tables

## 🔍 Key Analysis Components

### Data Investigation
- **Datasets**: `game_sales` (sales metadata) and `reviews` (critic & user scores)
- **Pre-computed helpers**: `users_avg_year_rating` and `critics_avg_year_rating` provide per-year average scores
- **Scope**: 400 rows per table (full Kaggle dataset has 13,000+ titles)

### Aggregation & Filtering Logic
- **Best sellers**: Sort `game_sales` by `games_sold` descending and take the top 10
- **Critic top years**: Join `game_sales` with `reviews`, group by `year`, require `COUNT(name) >= 4`, then rank by rounded average critic score
- **Golden years**: Combine the two pre-computed yearly tables and keep years where `avg_critic_score > 9` OR `avg_user_score > 9`

## 📈 Key Deliverables

### 1. `best_selling_games`
Top 10 highest-selling games with all columns from `game_sales`, ordered by `games_sold` descending.

| Column      | Description                          |
| ----------- | ------------------------------------ |
| `name`      | Name of the video game               |
| `platform`  | Gaming platform                      |
| `publisher` | Game publisher                       |
| `developer` | Game developer                       |
| `games_sold`| Number of copies sold (millions)     |
| `year`      | Release year                         |

### 2. `critics_top_ten_years`
Top 10 years with the highest average critic score, restricted to years with at least 4 games released.

| Column            | Description                                       |
| ----------------- | ------------------------------------------------- |
| `year`            | Release year                                      |
| `num_games`       | Number of games released that year                |
| `avg_critic_score`| Average critic score for the year (rounded to 2) |

Ordered by `avg_critic_score` descending.

### 3. `golden_years`
Years where critics and users broadly agreed that games released were highly rated (avg critic OR avg user score above 9).

| Column            | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `year`            | Release year                                         |
| `num_games`       | Number of games released that year                   |
| `avg_critic_score`| Average critic score for the year                    |
| `avg_user_score`  | Average user score for the year                      |
| `diff`            | `avg_critic_score - avg_user_score`                  |

Ordered by `year` ascending.

## 🛠️ Technologies Used
- **PostgreSQL**: Query execution
- **SQL**: `SELECT`, `JOIN`, `LEFT JOIN`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`, `COUNT`, `AVG`, `ROUND`, CTEs (`WITH`)

## 📁 Project Structure
```
golden-era-of-video-games/
├── README.md                              # This file
└── query/
    └── golden-era-of-video-games.sql      # Final analysis queries
```

## 🚀 Getting Started

### Prerequisites
- A running PostgreSQL instance with `game_sales`, `reviews`, `users_avg_year_rating`, and `critics_avg_year_rating` tables loaded
- SQL client (psql, DBeaver, DataGrip, or DataCamp's in-browser editor)

### Running the Queries
```sql
\i query/golden-era-of-video-games.sql
```

## 💡 Key Analysis Steps
1. **Best-selling titles**: Order `game_sales` by `games_sold` descending and `LIMIT 10`
2. **Critic top years**:
   - Join `game_sales` with `reviews` on `name`
   - Group by `year` and filter via `HAVING COUNT(name) >= 4`
   - Rank by rounded `AVG(critic_score)` descending and take the top 10
3. **Golden years**:
   - Join `users_avg_year_rating` with `critics_avg_year_rating` on `year`
   - Filter to years where either average score exceeds 9
   - Compute `diff = avg_critic_score - avg_user_score`
   - Sort by `year` ascending

## 🎯 Skills Demonstrated
- **Multi-table SQL joins**: `JOIN` and `LEFT JOIN` across sales and review datasets
- **Aggregation with thresholds**: `GROUP BY` + `HAVING` to enforce minimum sample size
- **Common Table Expressions**: Layered `WITH` clauses for readability
- **Set theory in filters**: `OR` predicates to combine independent quality signals
- **Output shaping**: Aliasing, rounding, and ordered presentation

## 📄 Data Sources

### `game_sales`
| Column      | Definition                          | Data Type |
| ----------- | ----------------------------------- | --------- |
| `name`      | Name of the video game              | varchar   |
| `platform`  | Gaming platform                     | varchar   |
| `publisher` | Game publisher                      | varchar   |
| `developer` | Game developer                      | varchar   |
| `games_sold`| Number of copies sold (millions)    | float     |
| `year`      | Release year                        | int       |

### `reviews`
| Column        | Definition                              | Data Type |
| ------------- | --------------------------------------- | --------- |
| `name`        | Name of the video game                  | varchar   |
| `critic_score`| Critic score according to Metacritic    | float     |
| `user_score`  | User score according to Metacritic      | float     |

### `users_avg_year_rating`
| Column          | Definition                                          | Data Type |
| --------------- | --------------------------------------------------- | --------- |
| `year`          | Release year of the games reviewed                  | int       |
| `num_games`     | Number of games released that year                  | int       |
| `avg_user_score`| Average user score of all games rated for the year | float     |

### `critics_avg_year_rating`
| Column            | Definition                                            | Data Type |
| ----------------- | ----------------------------------------------------- | --------- |
| `year`            | Release year of the games reviewed                    | int       |
| `num_games`       | Number of games released that year                    | int       |
| `avg_critic_score`| Average critic score of all games rated for the year | float     |

## 🔗 Related Work
- [Student Mental Health Analysis](../student-mental-health-analysis/)
- [Course Notes: Introduction to SQL](../../course-notes/introduction-to-sql/)
- [Course Notes: Intermediate SQL](../../course-notes/intermediate-sql/)

## 📈 Business Context
The global gaming market is projected to exceed $300 billion by 2027 (Mordor Intelligence). With massive incentives to produce the next hit, this analysis asks a simple question: are games getting better, or has the golden age of video games already passed?

---

**Project Status**: ✅ Completed
**Research Focus**: Cross-table SQL analysis of video game sales and critic/user reviews to identify a golden era of gaming
