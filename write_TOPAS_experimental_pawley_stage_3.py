import os
import glob
from pathlib import Path, PureWindowsPath
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT
    # ------------------------------------------------------------------------------------------------------
    phase_name = 'beta'
    sub_folder_1 = 'experimental_refinement'

    batchfile_folder = os.path.join(cwd, 'TOPAS', phase_name, 'batchfiles_topas')
    if not os.path.exists(batchfile_folder): os.makedirs(batchfile_folder)
    batchfile_path = os.path.join(batchfile_folder, f'pawley_refinement_of_experiment_stage_3.bat')

    exp_directory = os.path.join(cwd,'TOPAS', phase_name, 'experimental_refinement')
    topas_output_file = os.path.join(exp_directory, 'experimental_beta_pawley_stage_2.out' )

    stage_3_topas_input_file = os.path.join(exp_directory, 'experimental_beta_pawley_stage_3.inp')

    xy_out_path = os.path.join(exp_directory, 'exp_xy.xy')
    xy_out = f'Out_X_Ycalc("{xy_out_path}")\n'
    hkl_out_path = os.path.join(exp_directory, 'exp_hkl.dat')
    hkl_out = f'Create_d_Ip_file_dayton("{hkl_out_path}")\n\n'



    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    file_lines = []
    with open(topas_output_file, 'r') as f:
        file_lines = f.readlines()

    for i in range(len(file_lines)):
        if 'iters ' in file_lines[i]:
            file_lines[i] = '\titers 0\n'
    
        if 'continue_after_convergence' in file_lines[i]:
            file_lines[i] = '\t\'continue_after_convergence\n'
        
        if 'scale ' in file_lines[i]:
            file_lines[i] = '\t\tscale @ 1.0\n'

        if 'Out_X_Ycalc' in file_lines[i]:
            file_lines[i] = xy_out
        
        if 'Create_d_Ip' in file_lines[i]:
            file_lines[i] = hkl_out


    
    with open(stage_3_topas_input_file, 'w') as f:
        for i in range(len(file_lines)):
            f.write(file_lines[i])


    # write file for batch running topas pawley refinements of subgrouptree
    with open(batchfile_path, 'w') as f:
        f.write('cd C:\TOPAS6 \n')
        f.write(f'tc {stage_3_topas_input_file} \n')


if __name__ == "__main__":
    main()