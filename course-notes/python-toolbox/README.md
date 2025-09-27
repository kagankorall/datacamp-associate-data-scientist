# Python Toolbox

## Course Overview
This course covers advanced Python tools and techniques including iterators, list comprehensions, generators, and error handling.

## Key Topics Covered

### 1. Iterators and Iterables
- Understanding iteration
- Creating custom iterators
- Iterator protocol

### 2. List Comprehensions
- Basic list comprehensions
- Conditional comprehensions
- Nested comprehensions
- Dictionary and set comprehensions

### 3. Generators
- Generator functions
- Generator expressions
- Memory efficiency
- yield keyword

### 4. Error Handling
- try/except blocks
- Exception types
- Custom exceptions
- finally clause

## Key Concepts

### List Comprehensions
```python
# Basic list comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]

# Conditional list comprehension
even_squares = [x**2 for x in numbers if x % 2 == 0]

# Dictionary comprehension
square_dict = {x: x**2 for x in numbers}

# Nested list comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]
```

### Generators
```python
def fibonacci_generator(n):
    """Generate Fibonacci sequence up to n terms."""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Using generator
fib_gen = fibonacci_generator(10)
for num in fib_gen:
    print(num)

# Generator expression
squares_gen = (x**2 for x in range(10))
```

### Error Handling
```python
def safe_divide(a, b):
    """Safely divide two numbers with error handling."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Invalid input types!")
        return None
    finally:
        print("Division operation completed.")

# Usage
result = safe_divide(10, 2)
```

### Iterators
```python
class CountUp:
    """Custom iterator that counts up to a maximum value."""
    
    def __init__(self, max_value):
        self.max_value = max_value
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.max_value:
            self.current += 1
            return self.current
        else:
            raise StopIteration

# Usage
counter = CountUp(5)
for num in counter:
    print(num)
```

## Course Notes

## Introduction to Iterators

```python
for letter in 'DataCamp':
		print(letter)
```

Iterable: 

- Examples: Lists, strings, dictionaries, file connections
- An object with an associated iter() method
- Applying iter() to an iterable creates and iterator

Iterator:

- Produces next value with next()

```python
word = 'Da'
it = iter(word)
next(it) -> Result: 'D'
next(it) -> Result: 'a'
```

```python
word = 'Data'
it = iter(word)
print(*it)
#Result
D a t a
```

For dictionaries we should do the following:

```python
pythonistas = {'hugo': 'bowne-anderson', 'francis': 'castro'}
for key, value in pythonistas.items()
		print(key, value)
		
# Result
francis castro
hugo bowne-anderson
```

## enumerate()

```python
avengers = ['hawkeye','iron man','thor','quicksilver']
e = enumerate(avengers)
e_list = list(e)
print(e_list)
[(0,'hawkeye'),(1,'iron man'),(2,'thor'),(3,'quicksilver')]
```

We can iterate in enumerate by using index and value as follows:

```python
avengers = ['hawkeye','iron man','thor','quicksilver']
for index, value in enumerate(avengers):
		print(index, value)
```

## zip()

```python
avengers = ['hawkeye','iron man','thor','quicksilver']
names = ['barton','stark','odinson','maximoff']
z = zip(avengers,names)
z_list = list(z)
print(z_list)
[('hawkeye','barton'),('iron man','stark'),('thor','odinson'),('quicksilver','maximoff')]
```

## Loading data in chunks

There can be too much data to hold in memory → Solution: load data in chunks

```python
import pandas as pd
result = []
for chunk in pd.read_csv('data.csv', chunksize = 1000):
		result.append(sum(chunk['x']))
total = sum(result)
```

# List Comprehension

Instead of for loop, we can use list comprehension as follows:

```python
nums = [12,8,21,3,16]
new_nums = [num + 1 for num in nums]
```

This method collapse for loops for building lists into a single line. Components of this methods are iterable, iterator variable (represent members of iterable), output expression.

```python
# Nested for loop in list comprehension
pairs_2 = [(num1,num2) for num1 in range(0,2) for num2 in range (6,8)]
```

Readability is the drawback of this method instead of for loops.

Conditionals on the iterable:

```python
[num ** 2 for num in range(10) if num % 2 == 0]
```

## Generator Expressions

**List Comprehensions vs. Generators**: In Python, using different brackets changes the behavior of comprehensions significantly.

**Syntax Difference**:

- **List Comprehension**: Uses square brackets `[]` → `[x for x in range(10)]`
- **Generator Expression**: Uses parentheses `()` → `(x for x in range(10))`

**Key Differences**:

- **List Comprehension**: Returns a complete list object with all elements stored in memory
- **Generator Expression**: Returns a generator object that produces elements on-demand (lazy evaluation)

**Memory Usage**:

- List comprehensions create all elements immediately, consuming more memory
- Generators create elements one at a time, making them memory-efficient for large datasets

**Iteration**: Both are iterable, but generators can only be iterated once, while lists can be iterated multiple times.

**Use Cases**: Use generators for large datasets or when you only need to iterate once; use list comprehensions when you need to access elements multiple times or perform list operations.