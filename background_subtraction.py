import os
import glob
import torch
import pandas as pd
import utility_functions as util

def subtract_bkg_from_xy(xy_file, bkg_file):

    bragg = pd.read_csv(xy_file, sep='\s+', header = None, names = ['x','y'])
    bkg = pd.read_csv(bkg_file, sep='\s+', header = None, names = ['x','y'])

    sub = bragg['y'] - bkg['y']
    bragg['y'] = sub

    return bragg

cwd = os.getcwd()
material_name = 'wo3'

file_path = os.path.join(cwd, material_name, 'experimental_refinement', 'pawley')
xy_files = sorted(glob.glob(os.path.join(file_path, '*xy.xy')))
bkg_files = sorted(glob.glob(os.path.join(file_path, '*bkg.xy')))

for i,j in zip(xy_files, bkg_files):
    subtracted = subtract_bkg_from_xy(i,j)
    subtracted.to_csv(os.path.join(file_path, f'bkg_sub_{os.path.basename(i)}'), header=False, index=False, sep=' ')
    


