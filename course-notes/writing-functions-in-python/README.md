# Writing Functions in Python

## Course Overview
This course covers advanced function writing techniques in Python, including best practices, decorators, context managers, and functional programming concepts.

## Key Topics Covered

### 1. Advanced Function Concepts
- Function documentation and type hints
- Nested functions and closures
- Function factories
- Recursive functions

### 2. Decorators
- Understanding decorators
- Built-in decorators
- Writing custom decorators
- Decorator patterns

### 3. Context Managers
- Understanding context managers
- Using with statements
- Writing custom context managers
- contextlib module

### 4. Functional Programming
- Higher-order functions
- Map, filter, reduce
- Lambda functions advanced usage
- Partial functions

## Key Concepts

### Function Documentation and Type Hints
```python
from typing import List, Dict, Optional, Union, Callable

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        data (List[float]): List of numerical values
        
    Returns:
        Dict[str, float]: Dictionary containing mean, median, and std
        
    Raises:
        ValueError: If data list is empty
        TypeError: If data contains non-numeric values
        
    Examples:
        >>> calculate_statistics([1, 2, 3, 4, 5])
        {'mean': 3.0, 'median': 3.0, 'std': 1.58}
    """
    if not data:
        raise ValueError("Data list cannot be empty")
    
    if not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("All data elements must be numeric")
    
    import statistics
    return {
        'mean': statistics.mean(data),
        'median': statistics.median(data),
        'std': statistics.stdev(data) if len(data) > 1 else 0.0
    }
```

### Nested Functions and Closures
```python
def create_multiplier(factor: float) -> Callable[[float], float]:
    """Function factory that creates multiplier functions."""
    
    def multiplier(value: float) -> float:
        """Inner function that multiplies by the factor."""
        return value * factor
    
    return multiplier

# Usage
double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # Output: 10
print(triple(4))  # Output: 12

def outer_function(x):
    """Demonstrate closure behavior."""
    
    def inner_function(y):
        # Inner function has access to outer function's variables
        return x + y
    
    return inner_function

# Closure retains access to outer scope
add_five = outer_function(5)
result = add_five(3)  # Output: 8
```

### Decorators
```python
import functools
import time
from typing import Any, Callable

# Simple decorator
def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

# Decorator with parameters
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function execution on failure."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

# Usage
@timer
@retry(max_attempts=3, delay=0.5)
def risky_operation(x: int) -> int:
    if x < 0:
        raise ValueError("Negative values not allowed")
    return x ** 2

# Class-based decorator
class CountCalls:
    """Decorator to count function calls."""
    
    def __init__(self, func: Callable):
        self.func = func
        self.count = 0
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs) -> Any:
        self.count += 1
        print(f"{self.func.__name__} called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

### Context Managers
```python
from contextlib import contextmanager
import sqlite3

# Custom context manager class
class FileManager:
    """Context manager for file operations."""
    
    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing file: {self.filename}")
        if self.file:
            self.file.close()
        
        # Return True to suppress exceptions, False to propagate
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
        return False

# Using context manager
with FileManager('data.txt', 'w') as f:
    f.write("Hello, World!")

# Context manager using contextlib
@contextmanager
def database_connection(db_path: str):
    """Context manager for database connections."""
    conn = None
    try:
        print(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()
            print("Database connection closed")

# Usage
with database_connection('example.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
```

### Higher-Order Functions
```python
from functools import reduce, partial
from typing import Iterable, Callable

# Map, filter, reduce examples
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Map: Apply function to all elements
squared = list(map(lambda x: x**2, numbers))
squared_alt = [x**2 for x in numbers]  # List comprehension alternative

# Filter: Filter elements based on condition
evens = list(filter(lambda x: x % 2 == 0, numbers))
evens_alt = [x for x in numbers if x % 2 == 0]  # List comprehension alternative

# Reduce: Reduce sequence to single value
sum_all = reduce(lambda x, y: x + y, numbers)
product_all = reduce(lambda x, y: x * y, numbers)

# Custom higher-order functions
def apply_operations(data: Iterable, *operations: Callable) -> list:
    """Apply multiple operations to data sequentially."""
    result = data
    for operation in operations:
        result = operation(result)
    return result

def create_filter(condition: Callable) -> Callable:
    """Create a filter function based on condition."""
    return lambda data: filter(condition, data)

def create_mapper(transform: Callable) -> Callable:
    """Create a mapper function based on transformation."""
    return lambda data: map(transform, data)

# Usage
is_even = lambda x: x % 2 == 0
square = lambda x: x ** 2

result = apply_operations(
    numbers,
    create_filter(is_even),
    create_mapper(square),
    list
)
```

### Partial Functions
```python
from functools import partial

# Base function
def power(base: float, exponent: float) -> float:
    """Calculate base raised to exponent."""
    return base ** exponent

# Create specialized functions using partial
square = partial(power, exponent=2)
cube = partial(power, exponent=3)
square_root = partial(power, exponent=0.5)

# Usage
print(square(5))      # Output: 25
print(cube(3))        # Output: 27
print(square_root(16)) # Output: 4.0

# Partial with keyword arguments
def greet_user(greeting: str, name: str, punctuation: str = "!") -> str:
    return f"{greeting}, {name}{punctuation}"

# Create specialized greeting functions
say_hello = partial(greet_user, "Hello")
say_goodbye = partial(greet_user, "Goodbye", punctuation=".")

print(say_hello("Alice"))    # Output: Hello, Alice!
print(say_goodbye("Bob"))    # Output: Goodbye, Bob.
```

### Recursive Functions
```python
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Memoized version for efficiency
def fibonacci_memoized(n: int, memo: Dict[int, int] = None) -> int:
    """Memoized version of Fibonacci calculation."""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]

# Using lru_cache decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n: int) -> int:
    """Cached version using lru_cache decorator."""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

# Tree traversal example
def flatten_list(nested_list: List) -> List:
    """Recursively flatten a nested list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# Usage
nested = [1, [2, 3], [4, [5, 6]], 7]
flattened = flatten_list(nested)  # Output: [1, 2, 3, 4, 5, 6, 7]
```

## Course Notes

# Docstrings

![image.png](attachment:c6886045-7312-4041-b3e1-c737722bc849:image.png)

![image.png](attachment:9a016a61-e34a-4552-8a1b-f47f8e783cb8:image.png)

```python
# Example
def standardize(column):
  """Standardize the values in a column.

  Args:
    column (pandas Series): The data to standardize.

  Returns:
    pandas Series: the values as z-scores
  """
  # Finish the function so that it returns the z-scores
  z_score = (column - column.mean()) / column.std()
  return z_score

# Use the standardize() function to calculate the z-scores
df['y1_z'] = standardize(df.y1_gpa)
df['y2_z'] = standardize(df.y2_gpa)
df['y3_z'] = standardize(df.y3_gpa)
df['y4_z'] = standardize(df.y4_gpa)
```

# Context Managers

## Using context managers

A context manager:

- Sets up a context
- Runs your code
- Removes the context

```python
with open('my_file.txt') as my_file:
	text = my_file.read()
	length ) len(text)
print('The file is {} characters long'.format(length))
```

```python
# Example
# Open "alice.txt" and assign the file to "file"
with open('alice.txt') as file:
  text = file.read()

n = 0
for word in text.split():
  if word.lower() in ['cat', 'cats']:
    n += 1

print('Lewis Carroll uses the word "cat" {} times'.format(n))
```

```python
# Example 2
image = get_image_from_instagram()

# Time how long process_with_numpy(image) takes to run
with timer():
  print('Numpy version')
  process_with_numpy(image)

# Time how long process_with_pytorch(image) takes to run
with timer():
  print('Pytorch version')
  process_with_pytorch(image)
```

## Writing context managers

There are two ways to define a context manager:

1. class-based
2. function-based

There are five parts to define a context manager:

1. Define a function
2. Add any set up code your context needs
3. Use the yield keyword
4. Add any teardown code your context needs
5. Add the contextlib.contextmanager decorator

![image.png](attachment:bca5cd34-85a7-4ab4-bd00-2a5e3c4aacaa:image.png)

![image.png](attachment:5f4fe581-4770-441e-8aae-b3113f5ec096:image.png)

```python
# Example
# Add a decorator that will make timer() a context manager
@contextlib.contextmanager
def timer():
  """Time the execution of a context block.

  Yields:
    None
  """
  start = time.time()
  # Send control back to the context block
  yield None
  end = time.time()
  print('Elapsed: {:.2f}s'.format(end - start))

with timer():
  print('This should take approximately 0.25 seconds')
  time.sleep(0.25)
```

# Decorators

## Functions are objects

```python
# Example
# Add the missing function references to the function map
function_map = {
  'mean': mean,
  'std': std,
  'minimum': minimum,
  'maximum': maximum
}

data = load_data()
print(data)

func_name = get_user_input()

# Call the chosen function and pass "data" as an argument
function_map[func_name](data)
```

## Scope

## Closures