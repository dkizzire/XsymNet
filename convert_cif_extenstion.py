import os
import utility_functions as util

cwd = os.getcwd()

folder_path = os.path.join(cwd, 'wo3', 'subgroup_cifs_wo3')

old_extention = '.txt'
new_extention = '.cif'


def change_file_extensions_os(folder_path, old_extension, new_extension):
    for filename in os.listdir(folder_path):
        if filename.endswith(old_extension):
            base = filename[:-len(old_extension)]
            new_filename = base + new_extension
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            os.rename(old_file, new_file)

change_file_extensions_os(folder_path, old_extention, new_extention)