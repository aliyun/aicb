import json
import os
import argparse
import itertools
import re
from pathlib import Path
from datetime import datetime

def parse_value(s: str):
    """Try to parse a string into an integer, float, or return the original string."""
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            # Remove possible quotes
            if (s.startswith("'") and s.endswith("'")) or \
               (s.startswith('"') and s.endswith('"')):
                return s[1:-1]
            return s

def parse_value_range(value_str: str) -> list:
    """
    Parse the value string, supporting lists, ranges, and ranges with steps.
    - "[1,2,4,8]" -> [1, 2, 4, 8]
    - "[2-10]" -> [2, 3, 4, 5, 6, 7, 8, 9, 10]
    - "[2-10:2]" -> [2, 4, 6, 8, 10]
    - "value" -> ["value"]
    """
    value_str = value_str.strip()
    if not (value_str.startswith('[') and value_str.endswith(']')):
        return [parse_value(value_str)]

    inner_str = value_str[1:-1].strip()

    # Match ranges with steps, e.g.: 2-10:2
    range_step_match = re.fullmatch(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*:\s*(\d+)\s*$', inner_str)
    if range_step_match:
        start, end, step = map(int, range_step_match.groups())
        return list(range(start, end + 1, step))

    # Match simple ranges, e.g.: 2-10
    range_match = re.fullmatch(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*$', inner_str)
    if range_match:
        start, end = map(int, range_match.groups())
        return list(range(start, end + 1))

    # Default comma-separated list
    return [parse_value(x) for x in inner_str.split(',')]

def main():
    """Main function to parse arguments, generate config files and scripts."""
    parser = argparse.ArgumentParser(
        description="Generate JSON config files and execution scripts based on given parameter combinations.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-f', '--json_file',
        type=str,
        required=True,
        help="Path to the base JSON config file."
    )
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        required=True,
        help="Model name, used for creating output folders and in shell commands."
    )
    parser.add_argument(
        '-u', '--update',
        nargs=2,
        action='append',
        metavar=('KEY', 'VALUES'),
        required=True,
        help="""Attributes and their values to modify. Values can be a series in the following formats:
  - List: --update seq_length "[1,2,4,8]"
  - Range: --update world_size "[2-8]"
  - Range with step: --update micro_batch "[2-10:2]"
This parameter can be used multiple times to modify multiple attributes."""
    )
    args = parser.parse_args()

    # Read the base JSON file
    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{args.json_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{args.json_file}' is not valid JSON.")
        return

    # Create output directory
    root_out = Path("generated_configs")
    output_dir = root_out / args.model_name
    output_dir.mkdir(exist_ok=True)

    # Parse parameters to update
    update_params = {key: parse_value_range(val_str) for key, val_str in args.update}
    param_names = list(update_params.keys())
    param_values_lists = list(update_params.values())

    # Generate Cartesian product of all parameter values
    combinations = list(itertools.product(*param_values_lists))

    shell_commands = []
    
    # Iterate through all combinations
    for combo in combinations:
        new_data = base_data.copy()
        filename_parts = []
        
        # Update JSON data and construct filename
        for i, param_name in enumerate(param_names):
            value = combo[i]
            # Support nested keys, e.g. a.b.c
            keys = param_name.split('.')
            d = new_data
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value
            filename_parts.append(f"{param_name}_{value}")
            
        # Generate JSON filename and path
        output_filename_base = "-".join(filename_parts) + ".json"
        output_json_path = output_dir / output_filename_base

        # Write new JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)

        # Build shell command
        abs_config_path = Path.cwd() / output_json_path
        command = f"sh ./scripts/inference_workload_with_aiob.sh -m {args.model_name} -c {abs_config_path}"
        shell_commands.append(command)

    # Generate shell script with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shell_script_path = output_dir / f"run_all_{timestamp}.sh"
    with open(shell_script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Script generated by generate_configs.py at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(shell_commands) + "\n")
    
    # Make shell script executable
    os.chmod(shell_script_path, 0o755)

    print(f"Successfully generated {len(combinations)} config files and 'run_all_{timestamp}.sh' script.")
    print(f"Output directory: ./{output_dir}/")


if __name__ == '__main__':
    main()