import xsymnet_funcs as funcs
import os
import torch

cwd = os.getcwd()

# =============================================================================
# Example 1: Plot from a file
# =============================================================================
filepath = os.path.join(cwd, 'bso/experimental_refinement/beta_exp_xy.xy')
save_path = os.path.join(cwd, 'multichannel_plot_bso_beta_fromfile.png')

fig, axes = funcs.plot_multichannel_input(
    filepath=filepath,
    x_units='TOF (microseconds)',
    apply_smoothing=False,
    save_path=save_path
)
print('Example 1: Plot from file - Done!')


# =============================================================================
# Example 2: Plot from training data tensor
# =============================================================================
# Load training data (e.g., one random state from a subgroup)
training_file = os.path.join(cwd, 'bso/ml_data/beta_5/cif0152_randomstates_bragg.pt')
torch_data = torch.load(training_file)

# Get a single pattern (e.g., first random state)
y_data = torch_data[0]  # Shape: (seq_len,)
x_data = torch.arange(len(y_data))  # Use indices as x-axis

save_path2 = os.path.join(cwd, 'multichannel_plot_training_cif0152.png')

fig2, axes2 = funcs.plot_multichannel_input(
    x_data=x_data,
    y_data=y_data,
    x_units='Data Point Index',
    apply_smoothing=False,
    save_path=save_path2,
    title='Training Data: cif0152 (random state 0)'
)
print('Example 2: Plot from training tensor - Done!')


# =============================================================================
# Example 3: Plot with y_data only (x will be auto-generated as indices)
# =============================================================================
save_path3 = os.path.join(cwd, 'multichannel_plot_training_yonly.png')

fig3, axes3 = funcs.plot_multichannel_input(
    y_data=torch_data[5],  # 6th random state
    x_units='Index',
    apply_smoothing=True,
    sigma=2.0,
    save_path=save_path3,
    title='Training Data: cif0152 (random state 5, smoothed)'
)
print('Example 3: Plot with y_data only + smoothing - Done!')


# =============================================================================
# Helper function for zscore normalization
# =============================================================================
def zscore_1d(t, eps=1e-8):
    """Apply z-score normalization to a 1D tensor."""
    mean = t.mean()
    std = t.std()
    return (t - mean) / (std + eps)


# =============================================================================
# Example 4: Plot zscore of experiment data
# =============================================================================
# Load the experimental data
exp_x, exp_y = funcs.load_xy_data(filepath)
exp_y_zscore = zscore_1d(exp_y)

save_path4 = os.path.join(cwd, 'multichannel_plot_bso_beta_zscore.png')

fig4, axes4 = funcs.plot_multichannel_input(
    x_data=exp_x,
    y_data=exp_y_zscore,
    x_units='TOF (microseconds)',
    apply_smoothing=False,
    save_path=save_path4,
    title='Experiment Data: Z-Score Normalized'
)
print('Example 4: Plot zscore of experiment - Done!')


# =============================================================================
# Example 5: Plot zscore of training data
# =============================================================================
training_y_zscore = zscore_1d(torch_data[0])

save_path5 = os.path.join(cwd, 'multichannel_plot_training_zscore.png')

fig5, axes5 = funcs.plot_multichannel_input(
    y_data=training_y_zscore,
    x_units='Data Point Index',
    apply_smoothing=False,
    save_path=save_path5,
    title='Training Data: cif0152 (random state 0) Z-Score Normalized'
)
print('Example 5: Plot zscore of training data - Done!')


print('\nAll plots created successfully!')
