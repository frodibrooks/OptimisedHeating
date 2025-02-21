import json

def extract_junctions_and_reservoirs(features):
    """
    Extracts unique junctions and reservoirs from pipe start and end points.
    """
    junctions = set()
    reservoirs = set()
    coordinates = {}

    for feature in features:
        if feature['properties']['name'].startswith('Junction'): 
            junctions.add(feature['properties']['name'])
            coordinates[feature['properties']['name']] = [feature['properties']['x'], feature['properties']['y'], feature['properties']['z']]
        if feature['properties']['name'].startswith('Reservoir'): 
            reservoirs.add(feature['properties']['name'])
            coordinates[feature['properties']['name']] = [feature['properties']['x'], feature['properties']['y'], feature['properties']['z']]

    return list(junctions), list(reservoirs), coordinates


def geojson_to_epanet(geojson_data, inp_file):
    features = geojson_data['features']
    junctions, reservoirs, coordinates = extract_junctions_and_reservoirs(features)

    with open(inp_file, 'w') as file:
        # Write sections
        file.write("[TITLE]\n")
        file.write("Generated from GeoJSON\n\n")

        file.write("[JUNCTIONS]\n")
        file.write(";ID     Elevation       Demand      Pattern\n")
        for junction in junctions:
            # Extract demand if available from the GeoJSON data
            demand = 0  # Default demand if not found
            for feature in features:
                if feature['properties']['name'] == junction:
                    if 'DEMAND' in feature['properties']:
                        demand = feature['properties']['DEMAND']
                    break
            file.write(f"{junction}     {coordinates[junction][2]}       {demand} \n")
        file.write("\n")

        file.write("[RESERVOIRS]\n")
        file.write(";ID Head\n")
        for reservoir in reservoirs:
            file.write(f"{reservoir}        0 \n")
        file.write("\n")

        file.write("[PIPES]\n")
        file.write(";ID     Node1       Node2       Length      Diameter        Roughness       MinorLoss       Status\n")
        for feature in features:
            if feature['geometry']['type'] == 'LineString':
                properties = feature['properties']
                pipe_id = properties['name']
                start = properties['start']
                end = properties['end']
                length = properties['length']
                diameter = properties.get('extid:OD', '200')  # Default to 200 if not specified
                file.write(f"{pipe_id}      {start}     {end}       {length}        {diameter}      100     0       Open \n")
        file.write("\n")

        file.write("[COORDINATES]\n")
        file.write(";Node X-Coord Y-Coord\n")
        for key, value in coordinates.items():
            file.write(f"{key}      {value[0]}      {value[1]} \n")

    print(f"EPANET input file '{inp_file}' has been generated.")

# Example usage
geojson_file = 'Frodi_Vatnsendi.geojson'
with open(geojson_file) as file:
    geojson_data = json.load(file)

inp_file = 'output_epanet_model.inp'
geojson_to_epanet(geojson_data, inp_file)
