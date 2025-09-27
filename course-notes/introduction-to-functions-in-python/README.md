# Introduction to Functions in Python

## Course Overview
This course covers the fundamentals of creating and using functions in Python, including function syntax, parameters, return values, and best practices.

## Key Topics Covered

### 1. Function Basics
- Function definition and syntax
- Calling functions
- Function documentation

### 2. Parameters and Arguments
- Positional parameters
- Keyword arguments
- Default parameters
- Variable-length arguments (*args, **kwargs)

### 3. Return Values and Scope
- Return statements
- Local vs global scope
- Variable scope rules

## Key Concepts

### Basic Function Syntax
```python
def function_name(parameter1, parameter2):
    """
    Function docstring explaining what the function does.
    """
    # Function body
    result = parameter1 + parameter2
    return result

# Calling the function
output = function_name(5, 3)
print(output)  # Output: 8
```

### Default Parameters
```python
def greet(name, greeting="Hello"):
    """Greet someone with a customizable greeting."""
    return f"{greeting}, {name}!"

print(greet("Alice"))  # Output: Hello, Alice!
print(greet("Bob", "Hi"))  # Output: Hi, Bob!
```

### Variable Arguments
```python
def calculate_sum(*args):
    """Calculate sum of any number of arguments."""
    return sum(args)

def print_info(**kwargs):
    """Print key-value pairs."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Usage
total = calculate_sum(1, 2, 3, 4, 5)
print_info(name="Alice", age=25, city="New York")
```

### Lambda Functions
```python
# Lambda function for simple operations
square = lambda x: x ** 2
add = lambda x, y: x + y

# Using with built-in functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
```

## Course Notes

```python
def square(value): #Function header
		"""Returns the square of a value""" # Docstring
		new_value = value**2 # Function body
		print(new_value)
		return new_value
```

## Docstrings

Docstrings describe what your function does. Serve as documentation for your function. Placed in the immediate line after the function header. In between triple double quotes.

```python
def raise_to_power(value1, value2):
		"""Raise value1 to the power of value2"""
		new_value = value1 ** value2
		return new_value
```

Not all objects are accessible everywhere in a script.

## Scope

Part of the program where an object or name may be accessible

- Global scope - defined in the main body of a script
- Local scope - defined inside a function
- Built-in scope - names in the pre-defined built-ins module

In the example, new_value is a local scope.

## Nested Functions

```python
def mod2plus5(x1,x2,x3):
		"""Returns the remainder plus 5 of three values"""
		
		def inner(x):
				"""Return the remainder plus 5 of a values"""
				return x % 2 + 5
				
		return (inner(x1), inner(x2), inner(x3))
		
print(mod2plus5(1,2,3)) -> (6,5,6)
```

```python
def raise_val(n):
		"""Return the inner function"""
		
		def inner(x):
		"""Raise x to the power of n."""
		raised = x ** n
		return raised
		
		return inner
		
square = raise_val(2)
cube = raise_val(3)
print(square(2), cube(4)) -> 4 64
```

## Default Arguement

```python
def power(number, pow = 1):
		"""Raise number to the power of pow."""
		new_value = number ** pow
		return new_value
		
# Example
power(9,2) -> 81
power(9) -> 9
```

In the above code, pow is a default arguement

## Flexible Arguements (*args)

```python
def add_all(*args):
		"""Sum all values in *args together"""
		
		# Initialize sum
		sum_all = 0
		
		# Accumulate the sum
		for num in args:
				sum_all += num
		
		return sum_all

# Example
add_all(5,10,15,20) = 50
```

```python
def print_all(**kwargs):
		"""Print out key-value pairs in **kwargs"""
		
		# Print out the key-value pairs
		for key, value in kwargs.items():
				print(key + ": " + value)
				
# Example
print_all(name="dumbledore", job="headmaster")
```

## Lambda functions

```python
raise_to_power = lambda x,y : x ** y

# Example
raise_to_power(2,3) -> 8
```

```python
nums = [48, 6, 9, 21, 1]

square_all = map(lambda num: num ** 2, nums)

print(square_all) -> """ Map objesini döner"""

print(list(square_all)) -> [2304, 36, 81, 441, 1]
```

## Error Handling

```python
def sqrt(x):
		"""Returns the square root of a number"""
		if x < 0:
				raise ValueError('x must be non-negative')
		try:
				return x ** 0.5
		except TypeError:
				print('x must be an int of float')
```