import re

def modify_roughness(inp_file, output_file, new_roughness=0.4):
    with open(inp_file, 'r') as file:
        lines = file.readlines()

    in_pipes_section = False
    modified_lines = []

    for line in lines:
        # Detect when we enter the [PIPES] section
        if line.strip().startswith("[PIPES]"):
            in_pipes_section = True
            modified_lines.append(line)
            continue

        # Detect when we leave the [PIPES] section
        if in_pipes_section and line.strip().startswith("["):
            in_pipes_section = False

        # Modify only the lines in the [PIPES] section
        if in_pipes_section and not line.strip().startswith(";") and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                parts[5] = str(new_roughness)  # Roughness is the 5th column (index 4)
                line = "   ".join(parts) + "\n"

        modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

    print(f"Updated roughness of all pipes to {new_roughness} in {output_file}")

# Example usage
modify_roughness("change_rougness_2_04.inp", "modified_network.inp", new_roughness=0.4)
