import os
import numpy as np
import pandas as pd

# ======== Configuration ========
material_name = 'fe2o3'
input_folder = "data_bank_5"
output_folder = input_folder + "_truncated"
x_start = 3500.98585   # <-- Set your range here
x_end = 15260.31692     # <-- Set your range here
# ===============================
cwd = os.getcwd()
input_folder = os.path.join(cwd, material_name, input_folder)
output_folder = os.path.join(cwd, material_name, output_folder)
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    new_filename = os.path.splitext(filename)[0] + '.xy'
    output_path = os.path.join(output_folder, new_filename)

    try:
        # Load with whitespace splitting
        data = pd.read_csv(input_path, delim_whitespace=True, header=None)
        # use pandas to drop 3rd column if it exists

        # Drop third column if present
        if data.shape[1] == 3:
            data = data.drop(columns=[2])
        elif data.shape[1] != 2:
            raise ValueError(f"Unexpected number of columns: {data.shape[1]}")

        # Filter based on x range
        data.columns = ['x', 'y']

        # Filter based on x range
        data_filtered = data[(data['x'] >= x_start) & (data['x'] <= x_end)]

        data_filtered.to_csv(output_path, sep=' ', index=False, header=False, float_format='%.8f')
        print(f"✅ Processed: {filename}")
    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")