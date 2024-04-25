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
    batchfile_path = os.path.join(batchfile_folder, f'pawley_refinement_of_experiment_stage_2.bat')

    exp_directory = os.path.join(cwd,'TOPAS', phase_name, 'experimental_refinement')
    topas_output_file = os.path.join(exp_directory, 'experimental_beta_pawley_stage_1.out' )

    stage_2_topas_input_file = os.path.join(exp_directory, 'experimental_beta_pawley_stage_2.inp')

    xy_out_path = os.path.join(exp_directory, 'exp_xy.xy')
    xy_out = f'\'Out_X_Ycalc("{xy_out_path}")\n'
    hkl_out_path = os.path.join(exp_directory, 'exp_hkl.dat')
    hkl_out = f'\'Create_d_Ip_file_dayton("{hkl_out_path}")\n\n'

    endif_line = '#endif\n\n'


    # ------------------------------------------------------------------------------------------------------
    # END OF USER INPUT
    # ------------------------------------------------------------------------------------------------------

    file_lines = []
    with open(topas_output_file, 'r') as f:
        file_lines = f.readlines()
    
    new_lines_top = []
    new_lines_bottom = []

    for i in range(len(file_lines)):
        if 'iters ' in file_lines[i]:
            file_lines[i] = '\titers 500\n'
    
        if 'continue_after_convergence' in file_lines[i]:
            file_lines[i] = '\t\'continue_after_convergence\n'
        

        if 'load hkl' in file_lines[i]:
            break

        new_lines_top.append(file_lines[i])

    for i in range(len(file_lines)):
        if 'User defined macros' in file_lines[i]:
            for j in range(i,len(file_lines)):
                new_lines_bottom.append(file_lines[j])
            break
    
    with open(stage_2_topas_input_file, 'w') as f:
        for i in range(len(new_lines_top)):
            f.write(new_lines_top[i])
        f.write(xy_out)
        f.write(hkl_out)
        f.write(endif_line)
        for i in range(len(new_lines_bottom)):
            f.write(new_lines_bottom[i])

    # write file for batch running topas pawley refinements of subgrouptree
    with open(batchfile_path, 'w') as f:
        f.write('cd C:\TOPAS6 \n')
        f.write(f'tc {stage_2_topas_input_file} \n')


if __name__ == "__main__":
    main()