# FROM EXPERIMENTAL PAWLEY    
# TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 24.28584`,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -25.44401`)'
    # TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.143711169`, dsp, 11.64083`, dspsq, 11.71737`)'
    # background_line = 'bkg  @  1.67370339`  0.0515350436`  0.00972952795` -0.12759364`  0.20547493` -0.177201821`  0.116513928` -0.0589855793`  0.0192231893`  0.0184925627` -0.000494961725` -0.000104051593`'
    
    # if phase_name == 'alpha':
    #     experimental_data_file = f'hrp27188_bs_alpha_neutron.xye'
    #     TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs,-22.0105`_0.19111746,!difc_hrpd_bs, 48283.52311_1.67763, difa_hrpd_bs, -14.00371`_0.0876098944)'
    #     TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.168698361, dsp, 34.96508, dspsq, 2.83011)'
    # if phase_name == 'beta':
    #     experimental_data_file = f'Hrp27190_bs.xye'
    #     TOF_x_axis_calibration = 'TOF_x_axis_calibration(t0_hrpd_bs, 2.00496`_0.0846189708,!difc_hrpd_bs, 48255.09936_0.61297, difa_hrpd_bs, -3.22266`_0.0415199177)'
    #     TOF_sample_peakshape = 'tof_sample_peakshape(lor, 0.168698361, dsp, 34.96508, dspsq, 2.83011)'


        # get lattice information from cifs, use info to generate lines for hkl_Is section of topas input file
        # latparams, spacegroup_origin = funcs.get_cif_info(cif_files[i])
        # str_file = os.path.join(cwd, 'TOPAS', f'topas_strs_{phase_name}', f'topas{subgroup_nums[i]}.str')
        
        # generate pawley lattice parameter refinement lines - with or without vc macro
        # if use_vc_macro == True:
        #     if use_strain_modes == True:
        #         topas_latparam_lines,macro_lines = funcs.generate_latparam_lines_with_strain_modes_V2(str_file, spacegroup_origin, TOF_sample_peakshape)
        #     else:
        #         topas_latparam_lines = funcs.generate_lattice_parameter_lines_with_vc_macro(latparams, vc_matrix)
        #         macro_lines = []
        # else: 
        #     topas_latparam_lines = funcs.generate_pawley_lattice_lines_v2(latparams, spacegroup_origin)
        #     macro_lines = []


        # funcs.write_exp_refinement_strain_modes_stage_1(beamline_choice,
        #                                                 topas_input_file, 
        #                                                 use_vc_macro, 
        #                                                 TOF_x_axis_calibration, 
        #                                                 topas_latparam_lines,
        #                                                 macro_lines,
        #                                                 xdd_winpath, 
        #                                                 refinement_iterations, 
        #                                                 conv_criteria, 
        #                                                 start_finish_x,
        #                                                 background_line, 
        #                                                 xy_out_file=xy_out_file)





#------------------------------------
# FROM TOPAS ML STEP

            # mode_lines = funcs.generate_strain_and_symmetry_mode_lines_from_topas_str(str_files[i], 
            #                                                                           spacegroup_origin, 
            #                                                                           peakshape_lines, 
            #                                                                           random_strain[j], 
            #                                                                           custom_latparams=perturbed_latparams, 
            #                                                                           refine_beq = refine_beq, 
            #                                                                           beq_dict = beq_dict)
            
            # write the topas input file for cif i random_state j
            
            # funcs.write_inp_for_ML_with_strain_modes(beamline_choice,
            #                                          topas_input_file, 
            #                                          mode_lines, 
            #                                          start_finish_X, 
            #                                          xdd_winpath, 
            #                                          hkl_out_file=hkl_out_file, 
            #                                          xy_out_file=xy_out_file,
            #                                          dy_out_file=dy_out_file)