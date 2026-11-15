# Python Data Handling Cheatsheet

## Overview
This guide covers common Python patterns for working with files and data. It is intended for beginners building data and machine learning projects.

## Reading Files
Use the `with` statement so files close automatically:

```python
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

For CSV files, `pandas.read_csv("file.csv")` returns a DataFrame. For JSON, use `json.load(f)`.

## Lists and Dictionaries
- A **list** is an ordered, mutable collection: `nums = [1, 2, 3]`.
- A **dictionary** stores key-value pairs: `user = {"name": "Ammar", "city": "Amman"}`.
- List comprehension builds lists concisely: `squares = [x**2 for x in range(10)]`.

## Common Pandas Operations
| Task | Code |
|------|------|
| First rows | `df.head()` |
| Missing values | `df.isna().sum()` |
| Filter rows | `df[df["age"] > 18]` |
| Group and average | `df.groupby("city")["salary"].mean()` |
| Sort | `df.sort_values("score", ascending=False)` |

## Error Handling
```python
try:
    value = int(user_input)
except ValueError:
    print("Please enter a number.")
```

## Virtual Environments
Create one with `python -m venv venv`. Activate on Windows with `venv\Scripts\activate` and on macOS/Linux with `source venv/bin/activate`. Install packages with `pip install -r requirements.txt`.

## Best Practices
1. Use meaningful variable names.
2. Keep functions small and focused on one task.
3. Add type hints for clarity: `def add(a: int, b: int) -> int:`.
4. Never hard-code API keys; use environment variables.
