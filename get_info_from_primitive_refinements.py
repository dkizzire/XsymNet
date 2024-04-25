'''
Author: Dr. Dayton  Kizzire
Last Updated: 02/08/2024
Email: kizziredg@ornl.gov
'''

import os
import shutil
import json
import glob
import math
import numpy as np
from pathlib import Path, PureWindowsPath
from fractions import Fraction
import write_TOPAS_funcs as funcs


def main():
    cwd = os.getcwd()
    # ------------------------------------------------------------------------------------------------------
    # START OF USER INPUT  -- check all entries until 'END OF USER IMPUT'
    # ------------------------------------------------------------------------------------------------------
    
    phase_name = 'alpha'
    # directory structure : ./TOPAS/phase_name/sub_folder_1/sub_folder_2/
    sub_folder_1 = 'primitive_refinements'

    out_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', phase_name, sub_folder_1, '*.out')))
    filenames = [os.path.basename(x) for x in out_files]
    noext = [os.path.splitext(x)[0] for x in filenames]
    subgroup_nums = [x[9:13] for x in noext]
    print(subgroup_nums, '\n')
    
    in_files = sorted(glob.glob(os.path.join(cwd, 'TOPAS', phase_name, sub_folder_1, '*.inp')))

    cif_files = [os.path.join(cwd, 'subgroup_cifs', f'cif{subgroup_nums[x]}.cif') for x in range(len(subgroup_nums))]


    rwps = []
    latparams = []
    latparams_in = []
    t_matrix = []
    
    for i in range(len(out_files)):
        lats, space_group, origin, rwp = funcs.get_rwp_latparams_from_primitive_refinements(out_files[i])
        latparams.append(lats)
        rwps.append(rwp)
    
    for i in range(len(in_files)):
        lats, space_group, origin, rwp = funcs.get_rwp_latparams_from_primitive_refinements(in_files[i])
        latparams_in.append(lats)

    with open('./Primitive_Pawley_Refinements_info.txt', 'w') as f:
        f.write(f'Subgroup Label: Rwp Value \nPawley Refined Lattice Parameters \nStarting Lattice Parameters \nDifference \n\n')
        for i in range(len(out_files)):
            f.write(f'{subgroup_nums[i]} : Rwp = {rwps[i]} \n')
            f.write(f'{latparams[i][0]},  {latparams[i][1]},  {latparams[i][2]},  {latparams[i][3]},  {latparams[i][4]},  {latparams[i][5]}\n')
            f.write(f'{latparams_in[i][0]},  {latparams_in[i][1]},  {latparams_in[i][2]},  {latparams_in[i][3]},  {latparams_in[i][4]},  {latparams_in[i][5]}\n')
            f.write(f'{latparams[i][0] - latparams_in[i][0]},  {latparams[i][1] - latparams_in[i][1]},  {latparams[i][2] - latparams_in[i][2]},  {latparams[i][3] - latparams_in[i][3]},  {latparams[i][4] - latparams_in[i][4]},  {latparams[i][5] - latparams_in[i][5]}\n\n')





    for i in range(len(cif_files)):
        # extract the basis (P2C transformation matrix = pmat) and origin from isodistort cif files
        pmat, origin = funcs.get_basis_transformation_matirx(cif_files[i])

        # generate the inverse transformation matrix (qmat)
        qmat = np.linalg.inv(pmat)

        # convert pawley refined lattice parameters into column vector matrix
        clp = funcs.latparams2vectormatrix(latparams[i])
        # print(clp)

        # do clp dot qmat to generate parent lattice matrix -> take norm of each row to get parent lattice parameters
        plp_mat = np.matmul(clp, qmat)
        # print(plp_mat)

        # take the norm of each row of plp_mat to get the lattice parameters of the parent if the child had started with refined lattice parameters
        plp = [np.linalg.norm(plp_mat[i]) for i in range(len(plp_mat))]
        print(plp)
        plp_avg = np.average(plp)
        print(plp_avg)

        # copy parent cif to lattice subgroup directory // Open file and modify its lattice parameters
        parent_cif = os.path.join(cwd, 'subgroup_cifs', 'cif0001.cif')
        new_parent_dir = os.path.join(cwd, 'subgroup_cifs_lsg', f'{subgroup_nums[i]}')
        if not os.path.exists(new_parent_dir): os.makedirs(new_parent_dir)

        shutil.copy(parent_cif, new_parent_dir)
        new_parent_cif = os.path.join(new_parent_dir, 'cif0001.cif')
        basis_file = os.path.join(new_parent_dir, 'basis.txt')
        basis_json_filename = os.path.join(new_parent_dir, 'basis.json')
        cif_lines = []
        with open(new_parent_cif, 'r') as f:
            cif_lines = f.readlines()

        for i in range(len(cif_lines)):
            if '_cell_length_a' in cif_lines[i]:
                nline = cif_lines[i].strip().split()
                nline[1] = plp_avg
                print(nline)
                cif_lines[i]=f'{nline[0]}    {nline[1]}\n'
                print(cif_lines[i])

            if '_cell_length_b' in cif_lines[i]:
                nline = cif_lines[i].strip().split()
                nline[1] = plp_avg
                print(nline)
                cif_lines[i]=f'{nline[0]}    {nline[1]}\n'
                print(cif_lines[i])

            if '_cell_length_c' in cif_lines[i]:
                nline = cif_lines[i].strip().split()
                nline[1] = plp_avg
                print(nline)
                cif_lines[i]=f'{nline[0]}    {nline[1]}\n'
                print(cif_lines[i])

            if '_cell_volume' in cif_lines[i]: 
                nline = cif_lines[i].strip().split()
                nline[1] = (plp_avg**3)
                print(nline)
                cif_lines[i]=f'{nline[0]}    {nline[1]}\n'
                print(cif_lines[i])
                break
        
        with open(new_parent_cif, 'w') as f:
            for i in range(len(cif_lines)):
                f.write(cif_lines[i])
        
        # Note: isodistort prints basis vectors like so : basis={(b11, b12, b13), (b21, b22, b23), (b31, b32, b33)}  
                # however, (b11, b12, b13) is actually a column vector (Branton Campbell instructed me)
                # since python does math row majored the workflow is as follows:
                # python read isodistort basis as row majored -> transpose to column majored (in funcs.get_basis_transformation_matirx)
                # -> take inverse to get qmat (Child 2 Parent transformation matrix) 
                # -> transform refined child lattice parameters back to (refined) parent lattice parameters (apply qmat from the right, take norm of resultant rows)
                # -> take average of the new parent lattice parameters to get the new cubic lattice parameter
                # --> now for writing the basis vectors back to a file for isopydistort to read for automated (lattice subgroup) subgroup tree generation
                # --> transpose the pmat again to get it back in the form for isodistort to read it
                # --> write a json file for isopydistort to read the dict 
                # --> (also, isodistort uses a language that is 1 indexed, whereas python is 0 indexed)
                # ------> TL;DR python does math row major ordered, but to do the math, the basis vectors need to be column major ordered 
        pmat_t = pmat.T

        json_basis = {'subgroupsym':'1 P1 C1-1',
                      'pointgroupsym':'0',
                      'basis11':f'{Fraction(pmat_t[0,0])}',
                      'basis12':f'{Fraction(pmat_t[0,1])}',
                      'basis13':f'{Fraction(pmat_t[0,2])}',
                      'basis21':f'{Fraction(pmat_t[1,0])}',
                      'basis22':f'{Fraction(pmat_t[1,1])}',
                      'basis23':f'{Fraction(pmat_t[1,2])}',
                      'basis31':f'{Fraction(pmat_t[2,0])}',
                      'basis32':f'{Fraction(pmat_t[2,1])}',
                      'basis33':f'{Fraction(pmat_t[2,2])}'
                      }
        
        json_object = json.dumps(json_basis, indent=4)
        with open(basis_json_filename, 'w') as f:
            f.write(json_object)
        
        with open(basis_file, 'w') as f:
            f.write(f'a\'= {Fraction(pmat_t[0,0])} a + {Fraction(pmat_t[0,1])} b + {Fraction(pmat_t[0,2])} c \n')
            f.write(f'b\'= {Fraction(pmat_t[1,0])} a + {Fraction(pmat_t[1,1])} b + {Fraction(pmat_t[1,2])} c \n')
            f.write(f'c\'= {Fraction(pmat_t[2,0])} a + {Fraction(pmat_t[2,1])} b + {Fraction(pmat_t[2,2])} c \n')
            f.write(f'\n')
            f.write(f'# This is the basis to input into method 3 of isodistort to generate the subgroup tree section for this lattice subgroup')





if __name__ == "__main__":
    main()

