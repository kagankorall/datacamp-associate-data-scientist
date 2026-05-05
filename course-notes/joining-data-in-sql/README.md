# Joining Data in SQL

## Course Overview
A follow-up SQL course on joining tables, applying relational set theory, and writing subqueries. Examples use **PostgreSQL**.

## Key Topics Covered

### 1. Introducing Inner Joins
- `INNER JOIN` syntax with `ON`
- Joining on matching keys
- Table aliases for readability
- Joining multiple tables in a single query
- `USING` shortcut when join column names match

### 2. Outer Joins, Cross Joins and Self Joins
- `LEFT JOIN` / `RIGHT JOIN` / `FULL JOIN`
- Detecting unmatched rows via `NULL` checks
- `CROSS JOIN` for Cartesian products
- Self-joins for hierarchical / pairwise comparisons within a single table

### 3. Set Theory for SQL Joins
- `UNION` vs `UNION ALL`
- `INTERSECT` for shared rows across queries
- `EXCEPT` for rows in one query but not another
- Difference between set operations and joins

### 4. Subqueries
- Subqueries inside `WHERE` (scalar / list)
- `IN` and `NOT IN` with subqueries
- Subqueries inside `FROM` (derived tables)
- Subqueries inside `SELECT` (correlated)

## Key Concepts

### Inner join
```sql
SELECT c.name, p.population
FROM countries AS c
INNER JOIN populations AS p
  ON c.code = p.country_code;
```

### USING shortcut
```sql
SELECT *
FROM countries
INNER JOIN economies
  USING (code);
```

### Left join
```sql
SELECT c.name, p.population
FROM countries AS c
LEFT JOIN populations AS p
  ON c.code = p.country_code;
```

### Self join
```sql
SELECT p1.country, p1.year, p2.year, p1.size, p2.size
FROM populations AS p1
INNER JOIN populations AS p2
  ON p1.country = p2.country
 AND p1.year = p2.year - 5;
```

### Set operations
```sql
SELECT name FROM countries
UNION
SELECT name FROM cities;

SELECT code FROM countries
INTERSECT
SELECT country_code FROM economies;

SELECT code FROM countries
EXCEPT
SELECT country_code FROM economies;
```

### Subquery in WHERE
```sql
SELECT name
FROM countries
WHERE code IN (
  SELECT country_code
  FROM populations
  WHERE year = 2015 AND size > 50000000
);
```

### Subquery in FROM
```sql
SELECT continent, AVG(life_expectancy) AS avg_life
FROM (
  SELECT *
  FROM populations
  WHERE year = 2015
) AS pop_2015
GROUP BY continent;
```

## Skills Demonstrated

- Combining data from multiple related tables
- Choosing the right join type for the question being asked
- Comparing set operations vs joins for combining rows
- Decomposing complex queries with subqueries

## Key Takeaways

- **`INNER JOIN` keeps only matched rows**; outer joins preserve unmatched rows on one or both sides
- **`LEFT JOIN` + `IS NULL`** is the standard pattern for finding missing matches
- **`UNION` removes duplicates**, `UNION ALL` keeps them — `UNION ALL` is faster
- **`INTERSECT` / `EXCEPT`** compare full rows, not just join keys
- **Self-joins** require aliasing the same table twice
- **Subqueries** turn an intermediate result set into something you can filter or join against
