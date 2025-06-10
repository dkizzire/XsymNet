import os
import re
cwd = os.getcwd()


# This script modifies a TOPAS input file to set certain order parameters to be refined
# and others to be frozen at current value or set to zero. 
# The order parameters to be refined are specified in the 'refine_list'.
# Author: Dayton Kizzire - kizziredg@ornl.gov 
# Date: 2025-05-22

# ===========================================================
#^ Begin user-defined variables
# ===========================================================

refine_list = ['M1+']                   # List of order parameters to be refined ex: ['GM1+','L3-','X4+']
set_frozen_to_zero = True               # Set to True to set the distortion mode amplitude to zero for frozen order parameters
input_filename = '0036_dual.out'        # TOPAS file name to be modified
path = os.path.join(cwd, 'malcolm')     # Path to the directory containing the input file

# ===========================================================
#^ End of user-defined variables
# ===========================================================


# Input/output file paths
input_file = os.path.join(path, input_filename)
basename = os.path.splitext(input_filename)[0]

out_prefix = os.path.join(path, basename)
for i in range(len(refine_list)):
    if i != len(refine_list) - 1:
        out_prefix += '_'
    out_prefix += f'_{refine_list[i]}'

output_file = os.path.join(out_prefix + '.inp')

# loop to read the input file and modify the lines within the mode definitions section
in_mode_section = False
modified_lines = []

with open(input_file, 'r') as infile:
    for line in infile:
        stripped = line.strip()

        # Check for section start
        if '{{{mode definitions' in stripped:
            in_mode_section = True
            modified_lines.append(line)
            continue

        # Check for section end
        if '}}}' in stripped:
            in_mode_section = False
            modified_lines.append(line)
            continue

        if in_mode_section:
            split_line = line.split() # = ['prm','!a1','0.10105`','min','-1.41','max','1.41',"'I4/mmm[0,0,0]GM1+(a)[Mo1:e:dsp]",'A1(a)']
            param_name = split_line[1].strip('!')  # Remove the '!' character from the parameter name if it exists
            sym_info = split_line[7].split("]")
            op_info = sym_info[1].split("(")[0]   # Extract the order parameter information = 'GM1+'

            # Check if a order parameter is in the refine list
            if op_info in refine_list:
                new_line = f"\t\t{split_line[0]} {param_name} {split_line[2]} {split_line[3]} {split_line[4]} {split_line[5]} {split_line[6]} {split_line[7]} {split_line[8]}'\n"
            else:
                if set_frozen_to_zero:
                    new_line = f"\t\t{split_line[0]} !{param_name} 0.00000 {split_line[3]} {split_line[4]} {split_line[5]} {split_line[6]} {split_line[7]} {split_line[8]}'\n"
                else:
                    new_line = f"\t\t{split_line[0]} !{param_name} {split_line[2]} {split_line[3]} {split_line[4]} {split_line[5]} {split_line[6]} {split_line[7]} {split_line[8]}'\n"
            modified_lines.append(new_line)

        else:
            # Outside mode section — leave unchanged
            modified_lines.append(line)

# Write the modified content to a new output file
with open(output_file, 'w') as outfile:
    outfile.writelines(modified_lines)

print(f"Finished. Modified output written to: {output_file}")