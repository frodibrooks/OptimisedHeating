import json

def extract_junctions_and_reservoirs(features):
    junctions = set()
    reservoirs = set()
    coordinates = {}
    elevations = {}

    for feature in features:
        name = feature['properties']['name']
        coords = feature['geometry']['coordinates']
        
        # Handle junctions and reservoirs
        if name.startswith('Junction'):
            junctions.add(name)
            coordinates[name] = coords[:2]
            # Use the third coordinate as elevation (if available)
            elevations[name] = coords[2] if len(coords) > 2 else 0

        if name.startswith('Reservoir'):
            reservoirs.add(name)
            coordinates[name] = coords[:2]
            elevations[name] = coords[2] if len(coords) > 2 else 0

    return list(junctions), list(reservoirs), coordinates, elevations

def extract_vertices(features):
    vertices = {}
    
    for feature in features:
        if feature['geometry']['type'] == 'LineString':
            coordinates = feature['geometry']['coordinates']
            pipe_id = feature['properties']['name']
            
            if len(coordinates) > 2:
                vertices[pipe_id] = [coord[:2] for coord in coordinates[1:-1]]
    
    return vertices

def geojson_to_epanet(geojson_data, inp_file):
    features = geojson_data['features']
    junctions, reservoirs, coordinates, elevations = extract_junctions_and_reservoirs(features)
    vertices = extract_vertices(features)

    with open(inp_file, 'w') as file:
        file.write("[TITLE]\n")
        file.write("Generated from GeoJSON\n\n")

        file.write("[JUNCTIONS]\n")
        file.write(";ID     Elevation       Demand      Pattern\n")
        for junction in junctions:
            file.write(f"{junction}     {elevations[junction]}       0 \n")
        file.write("\n")

        file.write("[RESERVOIRS]\n")
        file.write(";ID Head\n")
        for reservoir in reservoirs:
            file.write(f"{reservoir}        {elevations[reservoir]} \n")
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
                diameter = properties.get('extid:OD', '200')
                file.write(f"{pipe_id}      {start}     {end}       {length}        {diameter}      100     0       Open \n")
        file.write("\n")

        file.write("[COORDINATES]\n")
        file.write(";Node Longitude Latitude\n")
        for key, value in coordinates.items():
            file.write(f"{key}      {value[0]}      {value[1]} \n")
        file.write("\n")
        
        file.write("[VERTICES]\n")
        file.write(";Link Longitude Latitude\n")
        for pipe_id, coords in vertices.items():
            for coord in coords:
                file.write(f"{pipe_id}      {coord[0]}      {coord[1]} \n")
        file.write("\n")
    
    print(f"EPANET input file '{inp_file}' has been generated.")

# Example usage
geojson_file = 'Frodi_Vatnsendi.geojson'
with open(geojson_file) as file:
    geojson_data = json.load(file)

inp_file = 'output_epanet_model.inp'
geojson_to_epanet(geojson_data, inp_file)
