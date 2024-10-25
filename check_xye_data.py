import pandas as pd

def check_data(file_path):
    try:
        # Read the file, assuming it has no header and is space-separated
        data = pd.read_csv(file_path, sep=r'\s+', header=None, names=['x', 'y', 'error'])

        # Check if the x values are unique
        x_values = data['x']
        unique = x_values.is_unique

        # Check if the x values are in ascending order
        ascending = x_values.is_monotonic_increasing

        return unique, ascending
    except UnicodeDecodeError as e:
        print(f"Error reading the file due to encoding issue: {e}")
        return None, None
    
# Usage example
file_path = './wo3/d8_03901_t1325.xye'  # Replace with your actual file path
unique, ascending = check_data(file_path)

if unique and ascending:
    print("The x column values are unique and in ascending order.")
elif not unique:
    print("The x column contains duplicate values.")
elif not ascending:
    print("The x column is not in ascending order.")

