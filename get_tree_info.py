import os
import write_TOPAS_funcs as funcs
import glob

cwd = os.getcwd()
material_name = 'wo3'
str_dir_name = 'topas_strs_wo3_cubic'
json_save_name = f'{material_name}_cubic_sg_tree_info.json'
strs = sorted(glob.glob(os.path.join(cwd, material_name, str_dir_name, '*')))
str_names = [os.path.splitext(os.path.basename(strs[i]))[0] for i in range(len(strs))]
strdict = {}
unique = []
for i in range(len(strs)):
    latparams, sgo = funcs.get_str_info(strs[i], 'strain')
    strdict[str_names[i]] = {'sg': sgo, 'lp': latparams}
    if latparams[0] not in unique:
        unique.append(latparams[0])
    if latparams[1] not in unique:
        unique.append(latparams[1])
    if latparams[2] not in unique:
        unique.append(latparams[2])
funcs.save_dict_to_json(strdict, os.path.join(cwd, material_name, json_save_name))
print('test')