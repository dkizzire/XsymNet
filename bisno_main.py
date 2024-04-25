import os
import json
import glob
import numpy as np
# import isopydistort
import isoget

# cwd = os.getcwd()
# selection_int = 1
# # psub_jsons = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_lsg', '*', '*.json')))
# psub_jsons = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_lsg', '0033', '*.json')))
# sub_dirs = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_lsg', '*' )))
# for i in range(len(psub_jsons)):
#     with open(psub_jsons[i], 'r') as f:
#         options = json.load(f)
#     parent_cif = os.path.join(sub_dirs[i], 'cif0001.cif')
#     outfile = os.path.join(sub_dirs[i], 'subgroup_tree.html')
#     tree = isopydistort.isoget.get(parent_cif, outfile, method=3, var_dict=options, isoformat='tree', selection=selection_int)
#     outfile = os.path.join(sub_dirs[i], 'subgroup_tree.str')
#     tree = isopydistort.isoget.get(parent_cif, outfile, method=3, var_dict=options, isoformat='topas', selection=selection_int)
#     outfile = os.path.join(sub_dirs[i], 'subgroup_tree.cif')
#     tree = isopydistort.isoget.get(parent_cif, outfile, method=3, var_dict=options, isoformat='structurefile', selection=selection_int)


cwd = os.getcwd()
selection_int = 1
# psub_jsons = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_lsg', '*', '*.json')))
psub_jsons = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_lsg', '0033', '*.json')))
sub_dirs = sorted(glob.glob(os.path.join(cwd, 'subgroup_cifs_lsg', '*' )))
for i in range(len(psub_jsons)):
    with open(psub_jsons[i], 'r') as f:
        options = json.load(f)
    parent_cif = os.path.join(sub_dirs[i], 'cif0001.cif')
    outfile = os.path.join(sub_dirs[i], 'subgroup_tree.html')
    tree = isoget.get(parent_cif, outfile, method=3, var_dict=options, isoformat='tree', selection=selection_int)
    outfile = os.path.join(sub_dirs[i], 'subgroup_tree.str')
    tree = isoget.get(parent_cif, outfile, method=3, var_dict=options, isoformat='topas', selection=selection_int)
    outfile = os.path.join(sub_dirs[i], 'subgroup_tree.cif')
    tree = isoget.get(parent_cif, outfile, method=3, var_dict=options, isoformat='structurefile', selection=selection_int)
