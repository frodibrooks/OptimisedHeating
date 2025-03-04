import re

def replace_nan(match):
    """Replace NaN values with 0."""
    num_str = match.group(0)
    
    # Handle NaN case
    if num_str.lower() == "nan":
        return "0"  # Replace NaN with 0
    
    return num_str  # Keep all other numbers unchanged

def process_epanet_inp(file_path, output_path):
    """Read an EPANET .inp file, replace NaN values, and save the formatted file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = []
    for line in lines:
        # Skip comment lines
        if line.strip().startswith(";") or line.strip() == "":
            updated_lines.append(line)
            continue
        
        # Use regex to find and replace NaN values
        updated_line = re.sub(r'\bNaN\b', replace_nan, line, flags=re.IGNORECASE)
        updated_lines.append(updated_line)

    with open(output_path, 'w') as file:
        file.writelines(updated_lines)

    print(f"Processed file saved to {output_path}")

# Example usage
input_file = "parse_me.inp"  # Replace with your file name
output_file = "VATNSENDI_with_demand_and_pumpstations_correct_parsed.inp"
process_epanet_inp(input_file, output_file)
