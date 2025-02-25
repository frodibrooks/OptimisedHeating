import re

def round_floats(match):
    """Round numbers to a max of 3 decimal places."""
    num_str = match.group(0)
    
    # Handle NaN case
    if num_str.lower() == "nan":
        return "0"  # Replace NaN with 0

    num = float(num_str)
    
    # Format with at most 3 decimal places
    return f"{num:.3f}".rstrip('0').rstrip('.')  # Removes unnecessary zeros (e.g., 1.500 -> 1.5)

def process_epanet_inp(file_path, output_path):
    """Read an EPANET .inp file, round floating-point numbers, replace NaN, and save the formatted file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = []
    for line in lines:
        # Skip comment lines
        if line.strip().startswith(";") or line.strip() == "":
            updated_lines.append(line)
            continue
        
        # Use regex to find and replace NaN values and round floating-point numbers
        updated_line = re.sub(r'nan|\d+\.\d+', round_floats, line, flags=re.IGNORECASE)
        updated_lines.append(updated_line)

    with open(output_path, 'w') as file:
        file.writelines(updated_lines)

    print(f"Processed file saved to {output_path}")

# Example usage
input_file = "round_this.inp"  # Replace with your file name
output_file = "VATNSENDI_with_demand_and_pumpstations.inp"
process_epanet_inp(input_file, output_file)
