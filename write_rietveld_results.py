import os 
import glob
import write_TOPAS_funcs as funcs
import pandas as pd

cwd = os.getcwd()
material_name = 'lxs'
phase_name = '6040'
out_files = sorted(glob.glob(os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP', '*.out')))

names = [os.path.basename(x) for x in out_files]
names = [os.path.splitext(x)[0] for x in names]

results = []
for i in range(len(out_files)):
    lat_params, space_group, herman, rwp = funcs.get_rwp_from_out(out_files[i])
    results.append([names[i], rwp, space_group, herman, lat_params])

df = pd.DataFrame(results)
df_sorted = df.sort_values(by=1)
df_sorted.to_csv(os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP', 'RESULTS.csv'), index=False)
print('check')
# with open(os.path.join(cwd, material_name, 'ml_data_generation', phase_name, 'RS_SGT_INP', 'RESULTS.txt'), 'w') as f:
#     f.write(f'{results[0], results[1], results[2], results[3], results[4]}\n\n')