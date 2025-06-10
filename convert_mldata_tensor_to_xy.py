import torch
import numpy as np
import os


def save_xy_files(base_xy_file, y_tensor, save_location, output_prefix="xy_output"):
    """
    Generates XY plottable data files using x-values from a base XY file and y-values from a tensor.
    
    Parameters:
        base_xy_file (str): Path to the base XY file containing x-values.
        y_tensor (torch.Tensor): A 2D tensor where each row is a separate plottable line.
        output_prefix (str): Prefix for output files.
    
    Each row of y_tensor will be written to a separate file with the x-values from base_xy_file.
    """
    # Load the base x values
    x_values = np.loadtxt(base_xy_file, usecols=[0])
    
    # Ensure tensor is on CPU and convert to numpy
    y_values = y_tensor.cpu().numpy()
    
    # Check if dimensions match
    if y_values.shape[1] != len(x_values):
        raise ValueError("Number of columns in y_tensor must match the number of x-values in the base XY file.")
    
    # Generate and save separate XY files
    for i, y_row in enumerate(y_values):
        output_filename = f"{output_prefix}_{i}.xy"
        out_file = os.path.join(save_location, output_filename)
        np.savetxt(out_file, np.column_stack((x_values, y_row)), fmt="%.6f", delimiter=" ")
        print(f"Saved: {output_filename}")

# Example Usage
if __name__ == "__main__":
    cwd = os.getcwd()
    material_name = 'bso'
    phase_name = 'alpha_3'
    # Sample x file (assume "base.xy" exists with x values) 
    base_xy_file = os.path.join(cwd, material_name, 'alpha_expt_manip.xy')
    y_tensor_file = os.path.join(cwd, material_name, 'ml_data', phase_name, 'topas0088_bso_alpha_3_rs.pt')
    y_tensor = torch.load(y_tensor_file)

    save_location = os.path.join(cwd, material_name, 'xy_files', phase_name)
    if not os.path.exists(save_location): os.makedirs(save_location)
    
    # Save the XY files
    save_xy_files(base_xy_file, y_tensor,save_location=save_location, output_prefix='0088')