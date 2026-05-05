# Intermediate SQL

## Course Overview
A hands-on follow-up to Introduction to SQL covering everything needed to start analyzing data with your own SQL code. Examples use **PostgreSQL**.

## Key Topics Covered

### 1. Selecting Data
- `SELECT` statements and column projection
- `COUNT()` for row counts
- Removing duplicates with `DISTINCT`
- Saving queries as `VIEW`s

### 2. Filtering Records
- `WHERE` with comparison operators (`=`, `<>`, `<`, `>`, `<=`, `>=`)
- Combining conditions with `AND`, `OR`, `NOT`
- Range filtering with `BETWEEN`
- Pattern matching with `LIKE` and wildcards (`%`, `_`)
- List membership with `IN`
- Handling `NULL` with `IS NULL` / `IS NOT NULL`

### 3. Aggregate Functions
- `SUM()`, `AVG()`, `MIN()`, `MAX()`, `COUNT()`
- Aliasing aggregate results with `AS`
- Arithmetic in `SELECT` (e.g. `budget - gross`)
- Rounding numerical results with `ROUND()`

### 4. Sorting and Grouping
- Sorting results with `ORDER BY` (ASC/DESC)
- Multi-column sorts
- Grouping rows with `GROUP BY`
- Filtering groups with `HAVING` (vs. `WHERE` on rows)

## Key Concepts

### Counting and distinct counts
```sql
SELECT COUNT(*) AS total_films
FROM films;

SELECT COUNT(DISTINCT country) AS unique_countries
FROM films;
```

### Filtering with WHERE
```sql
SELECT title, release_year
FROM films
WHERE release_year >= 2000
  AND country = 'USA';
```

### BETWEEN, IN, LIKE, NULL
```sql
SELECT title
FROM films
WHERE release_year BETWEEN 1990 AND 1999
  AND language IN ('English', 'Spanish')
  AND title LIKE 'The %'
  AND certification IS NOT NULL;
```

### Aggregates with aliases
```sql
SELECT
    AVG(budget)  AS avg_budget,
    MAX(gross)   AS max_gross,
    SUM(gross - budget) AS total_profit
FROM films;
```

### ROUND()
```sql
SELECT ROUND(AVG(budget), 2) AS avg_budget
FROM films;
```

### ORDER BY and GROUP BY with HAVING
```sql
SELECT release_year, AVG(gross) AS avg_gross
FROM films
GROUP BY release_year
HAVING AVG(gross) > 100000000
ORDER BY avg_gross DESC;
```

## Skills Demonstrated

- Filtering large tables down to relevant subsets
- Summarizing data with aggregate functions
- Grouping and ranking results for reporting
- Distinguishing row-level vs. group-level filters

## Key Takeaways

- **`WHERE` filters rows, `HAVING` filters groups** — they apply at different stages
- **`NULL` is not equal to anything** — use `IS NULL` / `IS NOT NULL`
- **Aggregates ignore `NULL`s** by default (except `COUNT(*)`)
- **`ORDER BY` runs after `GROUP BY`**, so you can sort by aggregated columns
- **Wildcards in `LIKE`**: `%` matches any sequence, `_` matches a single character
- **Use aliases liberally** to make aggregate output readable
