import os 
import glob
import write_TOPAS_funcs as funcs
import pandas as pd

cwd = os.getcwd()
material_name = 'fe2o3'
sub_folder = 'pawley_P1'
out_files = sorted(glob.glob(os.path.join(cwd, material_name, sub_folder, '*.out')))

names = [os.path.basename(x) for x in out_files]
names = [os.path.splitext(x)[0] for x in names]

results = []
for i in range(len(out_files)):
    #* rietveld refinement results
    # lat_params, space_group, rwp, hm_name = funcs.get_strainmode_latparams_from_topas_out(out_files[i])
    # results.append([names[i], rwp, space_group, hm_name, lat_params])

    # just rwp
    rwp = funcs.get_rwp_from_topas_out(out_files[i])
    results.append([names[i], rwp])


#* output for rietveld results
# df = pd.DataFrame(results)
# df_sorted = df.sort_values(by=1)
# df_sorted.to_csv(os.path.join(cwd, material_name, sub_folder, 'RESULTS.csv'), index=False)

#* output for just rwp
df_rwp = pd.DataFrame(results, columns=['Name', 'Rwp'])
df_rwp.to_csv(os.path.join(cwd, material_name, sub_folder, 'RWP_RESULTS.csv'), index=False)