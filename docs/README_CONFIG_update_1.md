# XSymNet Configuration File System

## Overview
This system allows you to manage training configurations through text files rather than editing the main script.

## Usage

### Basic Command
```bash
python -m training.xsymnet_bragg_seed_input <CONFIG_FILE> <SEED>
```

### Examples
```bash
# Train WO3 at 1043K with seed 48
python -m training.xsymnet_bragg_seed_input config_wo3_1043K.txt 48

# Train BSO alpha phase with seed 123
python -m training.xsymnet_bragg_seed_input config_bso_alpha.txt 123

# Use different seeds for reproducibility testing
python -m training.xsymnet_bragg_seed_input config_wo3_1043K.txt 42
python -m training.xsymnet_bragg_seed_input config_wo3_1043K.txt 100
```

## Configuration File Format

### General Rules
1. **Comments**: Lines starting with `#` are ignored
2. **Format**: `variable_name = value`
3. **Blank lines**: Ignored
4. **Inline comments**: Supported (text after `#` is ignored)

### Value Types Supported

#### Strings
```
material_name = wo3
phase_name = 1043K
cam_x_units_label = TOF (μs)
```

#### Raw Strings (preserves backslashes and special characters)
Use `r''` or `r""` prefix for LaTeX notation, regex patterns, or Windows paths:
```
# LaTeX notation for matplotlib legends
cam_legend_label = r'PbZr$_{0.1}$Ti$_{0.9}$O$_{3}$'
cam_legend_label = r'α-BiSn$_{2}$O$_{7}$'
cam_x_units_label = r'TOF ($\mu$s)'

# Windows paths (though forward slashes work everywhere)
path_to_data = r'C:\Users\Data\experiments'

# Regular expressions
file_pattern = r'\d{4}_sample_\w+\.dat'
```

#### Numbers (integers and floats)
```
epochs = 150
training_batchsize = 1024
percent_for_testing = 0.2
scale = 100
```

#### Booleans
```
train_model = True
save_model = False
do_zscore = True
```

#### Lists
```
scheduler_milestones = [40, 175]
scheduler_milestones = [50, 150, 180]
```

#### Special: SEED Placeholder
Use `SEED` in any string value and it will be replaced with the actual seed:
```
path_to_save_model = /path/to/model_seed_SEED.pt
# With seed 48, becomes: /path/to/model_seed_48.pt
```

### Example Configuration
```
# Material Information
material_name = wo3
phase_name = 1043K

# Training parameters
epochs = 150  # Number of training epochs
training_batchsize = 1024
confidence_iterations = 100
scheduler_milestones = [40, 175]

# Data configuration
data_start_index = 0
data_end_index = -1  # -1 means use all data
percent_for_testing = 0.2

# Model control
train_model = True
save_model = True
path_to_save_model = /models/wo3_seed_SEED.pt

# Experimental data
classify_experiment = True
experimental_files = wo3/experimental_p1s/*.xy

# Scaling
scale = 100
scaling_method = MinMax2d
do_zscore = True
```

## Configuration Variables

### Required Variables

#### Material and Phase
- `material_name`: Name of the material (string)
- `phase_name`: Phase identifier (string)

#### Training Parameters
- `epochs`: Number of training epochs (int)
- `training_batchsize`: Batch size for training (int)
- `confidence_iterations`: Number of confidence iterations (int)
- `scheduler_milestones`: Learning rate scheduler milestones (list)

#### Data Configuration
- `data_start_index`: Starting index for data slice (int, 0 for beginning)
- `data_end_index`: Ending index for data slice (int, -1 for all data)
- `percent_for_testing`: Fraction of data for testing (float, 0.0-1.0)

#### Model Control
- `train_model`: Whether to train the model (bool)
- `save_model`: Whether to save the trained model (bool)
- `path_to_save_model`: Path to save the model (string)

#### Experimental Classification
- `classify_experiment`: Whether to classify experimental data (bool)
- `experimental_files`: Path/glob pattern for experimental files (string)

#### Warmstart
- `warmstart_model`: Whether to use warmstart (bool)
- `warmstart_path`: Path to warmstart model (string)

#### Pretrained Model
- `load_pretrained_model`: Whether to load pretrained model (bool)
- `pretrain_path`: Path to pretrained model (string)

#### CAM (Class Activation Maps)
- `create_cam`: Whether to create CAMs (bool)
- `subgroup_cam_target`: Target subgroup for CAM (string)
- `cam_save_basename`: Base name for saved CAM files (string)
- `cam_x_units_label`: X-axis label for CAM plots (string)
- `cam_legend_label`: Legend label for CAM plots (string)

#### Scaling
- `scale`: Scaling factor (int)
- `scaling_method`: Scaling method name (string: 'MinMax2d', 'MaxAbs', 'None')
- `do_zscore`: Whether to apply z-score normalization (bool)

## Tips

1. **Create multiple configs**: Make separate config files for different experiments
   - `config_wo3_1043K.txt`
   - `config_wo3_1085K.txt`
   - `config_bso_alpha.txt`

2. **Use descriptive names**: Include material and phase in config filename

3. **Comment your changes**: Add comments explaining non-standard settings

4. **Version control**: Keep config files in version control alongside code

5. **Batch experiments**: Use shell scripts to run multiple configurations:
   ```bash
   for seed in 42 48 123 456; do
       python -m training.xsymnet_bragg_seed_input config_wo3_1043K.txt $seed
   done
   ```

## File Glob Patterns

The `experimental_files` variable supports glob patterns:
```
# Single file
experimental_files = wo3/experimental_p1s/P1_wo3_1085K_xy.xy

# All .xy files in directory
experimental_files = wo3/experimental_p1s/*.xy

# All files matching pattern
experimental_files = wo3/experimental_p1s/P1_*_1085K_*.xy
```

## Error Handling

The configuration reader will:
- **Warn** about malformed lines (missing `=`)
- **Warn** if no experimental files match the pattern
- **Exit** if config file not found
- **Exit** if seed is not a valid integer

## Troubleshooting

### "Configuration file not found"
- Check that the path to the config file is correct
- Use absolute path or path relative to where you run the script

### "No files found matching pattern"
- Check the glob pattern in `experimental_files`
- Verify files exist in the specified directory
- Remember paths are relative to the script's working directory

### Variables not being set
- Check spelling of variable names (must match exactly)
- Ensure proper format: `variable_name = value`
- Check for syntax errors in lists or complex values
