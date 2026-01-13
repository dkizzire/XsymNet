import ast
import os
import glob

def read_config_file(config_path, seed_value):
    """
    Reads a configuration file and returns a dictionary of variable assignments.
    
    Supports:
    - Comments (lines starting with #)
    - Blank lines
    - String values (with or without quotes)
    - Numeric values (int, float)
    - Boolean values (True/False)
    - Lists (e.g., [1, 2, 3])
    - None values
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Dictionary mapping variable names to their values
    """
    config_vars = {}
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for variable assignment
            if '=' not in line:
                print(f"Warning: Line {line_num} does not contain '=' and will be skipped: {line}")
                continue
            
            # Split on first '=' only
            parts = line.split('=', 1)
            if len(parts) != 2:
                print(f"Warning: Line {line_num} has invalid format and will be skipped: {line}")
                continue
            
            var_name = parts[0].strip()
            var_value_str = parts[1].strip()
            
            # Remove inline comments (but be careful with strings containing #)
            # Only remove # if it's outside quotes
            if '#' in var_value_str:
                # Simple approach: if value starts with quote, don't strip comments
                if not (var_value_str.startswith('"') or var_value_str.startswith("'")):
                    var_value_str = var_value_str.split('#')[0].strip()
            
            # Parse the value
            try:
                # Check if it's a raw string (starts with r' or r")
                if var_value_str.startswith(("r'", 'r"', "R'", 'R"')):
                    # Handle raw string - remove the r prefix and quotes
                    quote_char = var_value_str[1]  # Get the quote character (' or ")
                    if var_value_str.endswith(quote_char):
                        # Extract content between quotes, preserving backslashes
                        var_value = var_value_str[2:-1]
                    else:
                        raise ValueError(f"Unclosed raw string: {var_value_str}")
                else:
                    # Try to evaluate as Python literal (handles int, float, bool, list, None, etc.)
                    var_value = ast.literal_eval(var_value_str)
            except (ValueError, SyntaxError):
                # If that fails, treat as string (remove quotes if present)
                var_value = var_value_str.strip('"').strip("'")
            
            config_vars[var_name] = var_value
    
    for var_name, var_value in config_vars.items():
        # Special handling for paths with SEED, MATNAME, or PNAME placeholders
        if isinstance(var_value, str) and 'SEED' in var_value:
            var_value = var_value.replace('SEED', str(seed_value))
        if isinstance(var_value, str) and 'MATNAME' in var_value:
            var_value = var_value.replace('MATNAME', config_vars.get('material_name', 'NO_MAT_NAME'))
        if isinstance(var_value, str) and 'PNAME' in var_value:
            var_value = var_value.replace('PNAME', config_vars.get('phase_name', 'NO_PHASE_NAME'))
            
        config_vars[var_name] = var_value
               

    return config_vars


def apply_config_to_globals(config_dict, globals_dict, seed_value):
    """
    Applies configuration dictionary to global variables.
    Also handles special cases like experimental_files glob pattern and seed substitution.
    
    Args:
        config_dict (dict): Dictionary of configuration variables
        globals_dict (dict): The globals() dictionary to update
        seed_value (int): The seed value to substitute in paths
    """
    cwd = os.getcwd()
    
    for var_name, var_value in config_dict.items():
        # Special handling for experimental_files
        if var_name == 'experimental_files':
            # Handle glob pattern
            file_pattern = var_value
            # If it's not an absolute path, make it relative to cwd
            if not os.path.isabs(file_pattern):
                file_pattern = os.path.join(cwd, file_pattern)
            
            experimental_files = sorted(glob.glob(file_pattern))
            if not experimental_files:
                print(f"Warning: No files found matching pattern: {file_pattern}")
                experimental_files = []
            
            globals_dict['experimental_files'] = experimental_files
            
            # Also create experimental_names
            experimental_names = [os.path.splitext(os.path.basename(f))[0] for f in experimental_files]
            globals_dict['experimental_names'] = experimental_names
        
        # Special handling for paths with SEED placeholder
        elif isinstance(var_value, str) and 'SEED' in var_value:
            var_value = var_value.replace('SEED', str(seed_value))
            globals_dict[var_name] = var_value
               
        elif isinstance(var_value, str) and 'MATNAME' in var_value:
            var_value = var_value.replace('MATNAME', config_dict.get('material_name', 'NO_MAT_NAME'))
            globals_dict[var_name] = var_value

        elif isinstance(var_value, str) and 'PNAME' in var_value:
            var_value = var_value.replace('PNAME', config_dict.get('phase_name', 'NO_PHASE_NAME'))
            globals_dict[var_name] = var_value
        
        else:
            # Normal assignment
            globals_dict[var_name] = var_value
    
    print(f"\n{'='*60}")
    print(f"Configuration loaded successfully from config file")
    print(f"{'='*60}")
    print(f"Material: {globals_dict.get('material_name', 'N/A')}")
    print(f"Phase: {globals_dict.get('phase_name', 'N/A')}")
    print(f"Epochs: {globals_dict.get('epochs', 'N/A')}")
    print(f"Batch Size: {globals_dict.get('training_batchsize', 'N/A')}")
    print(f"Seed: {seed_value}")
    print(f"Experimental files found: {len(globals_dict.get('experimental_files', []))}")
    print(f"{'='*60}\n")