# Introduction to SQL

## Course Overview
A short introductory course on creating and querying relational databases with SQL. Examples in this course use **PostgreSQL** as the database system.

## Key Topics Covered

### 1. Relational Databases
- What a relational database is and why it's used
- Tables, rows (records), columns (fields), and primary keys
- Common data types (`INT`, `VARCHAR`, `DATE`, etc.)
- Database vs. spreadsheet differences
- SQL flavors and dialect differences (PostgreSQL, MySQL, SQL Server)

### 2. Querying
- Selecting columns with `SELECT`
- Aliasing columns with `AS`
- Returning unique values with `DISTINCT`
- Creating reusable views with `CREATE VIEW`
- Limiting results with `LIMIT`

## Key Concepts

### Basic SELECT
```sql
SELECT name, release_year
FROM films;
```

### Selecting all columns
```sql
SELECT *
FROM films;
```

### Aliasing
```sql
SELECT title AS film_title
FROM films;
```

### Distinct values
```sql
SELECT DISTINCT country
FROM films;
```

### Creating a view
```sql
CREATE VIEW films_2020 AS
SELECT title, director
FROM films
WHERE release_year = 2020;
```

### Limiting rows
```sql
SELECT *
FROM films
LIMIT 10;
```

## Skills Demonstrated

- Reading data from a relational database with PostgreSQL
- Writing simple, readable `SELECT` queries
- Using aliases and views to make queries reusable
- Reasoning about table structure and data types

## Key Takeaways

- **Relational databases store data in tables** linked by keys
- **SQL is declarative**: describe what you want, not how to get it
- **`DISTINCT` removes duplicates** in the returned result set
- **Views save queries**, not data — they always reflect the current table contents
- **Dialects differ slightly** but core SQL syntax is portable across systems
