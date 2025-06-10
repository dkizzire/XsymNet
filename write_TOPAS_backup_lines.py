        #*----------------------------------------------------------------------------------------------------------------------------
        #* Get optimized scale
        #*----------------------------------------------------------------------------------------------------------------------------

        # scale_input_file = os.path.join(input_path, f'scale_{str_name}.inp')
        # scale_output_file = os.path.join(input_path, f'scale_{str_name}.out')

        # mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str_v2(str_file, spacegroup_origin, random_strain[0], beq_dict=beq_dict, refine_strain=True, refine_distortion=True)
        # # insert peakshape, r_bragg, and scale into mode lines
        # mode_lines.insert(4, f'\t\tphase_name {str_name}\n')
        # mode_lines.insert(4, f'{input_dict["peakshape_lines"]}\n')
        # mode_lines.insert(4, f'\t\tr_bragg 0.0\n')
        # mode_lines.insert(4, f'\t\tscale @ 1.0\n')

        # # write scale optimization input file
        # with open(scale_input_file, 'w') as f:
        #     f.write(input_dict['scale_refinement_lines'])
        #     f.write(input_dict['global_lines'])
        #     f.write(input_dict['xdd_lines'])
        #     f.write(input_dict['instrument_lines'])
        #     for k in range(len(mode_lines)):
        #         f.write(mode_lines[k])

        # # Convert topas input path to windows so TOPAS can run it (TOPAS is Windows only)
        # scaling_winpath = PureWindowsPath(Path(scale_input_file).absolute()) 

        # # write a windows batchfile to run all random_states with tc.exe (topas command line)
        # batchfile_path = os.path.join(input_path, 'scaling.bat')
        # with open(batchfile_path, 'w') as f:
        #     f.write(f'cd {path_to_topas_directory} \n')
        #     f.write(f'tc "{scaling_winpath}" \n')
        # batchfile_path = PureWindowsPath(Path(batchfile_path).absolute())

        # # Run the batch file and capture output and errors
        # result = subprocess.run(batchfile_path, shell=False, capture_output=True, text=True)

        # # grab the optimized scale from output file
        # scale = 1.0
        # with open(scale_output_file, 'r') as f:
        #     lines = f.readlines()
        #     for i in range(len(lines)):
        #         if 'scale @ ' in lines[i]:
        #             scale = float(lines[i].strip().split('@')[1].strip().strip('`'))
        #             break 
                
        # print(f'{str_name} scale = {scale}\n')
        # shutil.copy(scale_input_file, os.path.join(scale_file_destination, os.path.basename(scale_input_file)))
        # shutil.copy(scale_output_file, os.path.join(scale_file_destination, os.path.basename(scale_output_file)))