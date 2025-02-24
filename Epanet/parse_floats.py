import re

def round_floats(match):
    """Round numbers to a max of 3 decimal places."""
    num = float(match.group(0))
    
    # Format with at most 3 decimal places
    return f"{num:.3f}".rstrip('0').rstrip('.')  # Removes unnecessary zeros (e.g., 1.500 -> 1.5)

def process_epanet_inp(file_path, output_path):
    """Read an EPANET .inp file, round floating-point numbers, and save the formatted file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = []
    for line in lines:
        # Skip comment lines
        if line.strip().startswith(";") or line.strip() == "":
            updated_lines.append(line)
            continue
        
        # Use regex to find and round floating-point numbers
        updated_line = re.sub(r'\d+\.\d+', round_floats, line)
        updated_lines.append(updated_line)

    with open(output_path, 'w') as file:
        file.writelines(updated_lines)

    print(f"Processed file saved to {output_path}")

# Example usage
input_file = "change_floating_point_numb.inp"  # Replace with your file name
output_file = "network_rounded.inp"
process_epanet_inp(input_file, output_file)
