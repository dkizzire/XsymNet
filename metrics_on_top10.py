import os
import numpy as np
import pandas as pd
import glob
import xsymnet_funcs as xf

cwd = os.getcwd()

material = 'bso'
files = sorted(glob.glob(os.path.join(cwd, material, 'zzz_bso_alpha_5_MinMax-Zscale_seed*')))
str_files = sorted(glob.glob(os.path.join(cwd, material, f'topas_strs_{material}', '*' )))

seeded_predictions = xf.get_top10_output(files)
# print(seeded_predictions)
stability_df = xf.analyze_prediction_stability(seeded_predictions, top_n=10, str_files=str_files)

stability_df.to_csv(os.path.join(cwd, material, 'prediction_stability_top10_alpha_5.csv'), index=False)