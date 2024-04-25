import os
import shutil
from zipfile import ZipFile 
import glob
import utility_functions as funcs

cwd = os.getcwd()
phase_name = 'alpha'

dirs = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_lsg', '*')))
cifzip = 'cif.zip'
strzip = 'topas.zip'

cif_name = 'cifs'
str_name = 'topas_strs'

cifdirs = [os.path.join(x, cifzip) for x in dirs]
strdirs = [os.path.join(x, strzip) for x in dirs]

newcifdirs = [os.path.join(x, cif_name) for x in dirs]
newstrdirs = [os.path.join(x, str_name) for x in dirs]


# for i in range(len(cifdirs)):
#     funcs.pyunzip(cifdirs[i], newcifdirs[i])
#     funcs.pyunzip(strdirs[i], newstrdirs[i])


subgroup_nums = [(os.path.basename(os.path.normpath(x))) for x in dirs]
subgroup_nums_ints = [int(x) for x in subgroup_nums]
diff = [0]
for i in range(1,len(subgroup_nums_ints)):
    diff.append(subgroup_nums_ints[i]-subgroup_nums_ints[i-1])

print(diff)
print(len(diff))
diffsum = sum(diff) +33
print(diffsum)
# put in an assert statement that checks if len(subgroup_cifs) ((original subgroup tree)) equals diffsum

# extract the last diff[i] elements of each subgroup directory in subgroup_cifs_lfg/0xxx/cifs & subgroup_cifs_lfg/0xxx/topas_strs

section_cifs = []
section_strs = []
for i in range(len(dirs)):
    print(f'LATTICE SUBGROUP # {i+1}')
    subgroup_cifs = sorted(glob.glob(os.path.join(newcifdirs[i], '*.txt')))
    subgroup_strs = sorted(glob.glob(os.path.join(newstrdirs[i], '*.str')))
    print(len(subgroup_cifs), ' # cifs in section subgroup tree')
    print(len(subgroup_strs), ' # strs in section subgroup tree')
    if i == 0:
        for j in range(len(subgroup_cifs)):
            section_cifs.append(subgroup_cifs[j])
            section_strs.append(subgroup_strs[j])
        print(len(section_cifs), ' section_cifs')
        print(len(section_strs), ' section_strs')
    else:
        subgroup_cifs_cut = subgroup_cifs[-diff[i]:]
        subgroup_strs_cut = subgroup_strs[-diff[i]:]
        print(len(subgroup_cifs_cut), f' # of cifs in lattice subgroup section {i}')
        print(len(subgroup_strs_cut), f' # of strs in lattice subgroup section {i}')
        for k in range(len(subgroup_cifs_cut)):
            section_cifs.append(subgroup_cifs_cut[k])
            section_strs.append(subgroup_strs_cut[k])
        print(len(section_cifs), ' section_cifs in aggregated total')
        print(len(section_strs), ' section_strs in aggregated total')

print()
print(len(section_cifs), ' = # of cifs in sectioned list')
print(len(section_strs), ' = # of strs in sectioned list')

# get original subgroup names from subgroup_cifs
og_tree_names = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs', '*')))
og_tree_names = [os.path.basename(os.path.normpath(x)) for x in og_tree_names]
og_strs_names = sorted(glob.glob(os.path.join(cwd, 'TOPAS', 'topas_strs', '*')))
og_strs_names = [os.path.basename(os.path.normpath(x)) for x in og_strs_names]

# make new directories to store sectioned cifs and strs
section_cif_dir = os.path.join(cwd, phase_name, 'subgroup_cifs_sectioned')
section_str_dir = os.path.join(cwd, 'TOPAS', phase_name, 'topas_strs_sectioned')
if not os.path.exists(section_cif_dir): os.makedirs(section_cif_dir)
if not os.path.exists(section_str_dir): os.makedirs(section_str_dir)
# make filenames with new paths for sectioned cifs and strs
new_cif_filenames = [os.path.join(section_cif_dir, x) for x in og_tree_names]
new_str_filenames = [os.path.join(section_str_dir, x) for x in og_strs_names]
# copy sectioned cifs and strs into composite directory
for i in range(len(section_cifs)):
    dest_cif = shutil.copy(section_cifs[i], new_cif_filenames[i])
    dest_str = shutil.copy(section_strs[i], new_str_filenames[i])
