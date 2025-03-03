import geopandas as gpd
import pandas as pd

def extract_junctions_and_reservoirs(gdf):
    """
    Extracts unique junctions and reservoirs from the GeoPackage.
    """
    junctions = set()
    reservoirs = set()
    coordinates = {}
    demands = {}

    for _, row in gdf.iterrows():
        name = row['name']
        x, y, z = row['geometry'].x, row['geometry'].y, row.get('z', 0)

        if name.startswith('Junction'): 
            junctions.add(name)
            coordinates[name] = [x, y, z]
            # print(row.get('DAILY_ENERGY'))
            print(row.get('DEMAND'))
            if row.get('DEMAND') == 0.0:
                if row.get('DAILY_ENERGY', 0.0) == 0.0:
                    demands[name] = 0
                else:
                    demands[name] = ((row['DAILY_ENERGY'] * 1000) / (984.4778019 * 4.18176019 * 45))/24
            else:
                demands[name] = row['DEMAND']

        elif name.startswith('Reservoir'): 
            reservoirs.add(name)
            coordinates[name] = [x, y, z]

    return list(junctions), list(reservoirs), coordinates, demands

def geopackage_to_epanet(gpkg_file, inp_file, junction_layer, pipe_layer, reservoir_layer):
    """
    Converts a GeoPackage to an EPANET INP file.
    """
    layers = gpd.read_file(gpkg_file, layer=None)  # None returns all layer names
    print("Available layers:", layers)
    junctions_gdf = gpd.read_file(gpkg_file, layer=junction_layer)
    pipes_gdf = gpd.read_file(gpkg_file, layer=pipe_layer)
    reservoirs_gdf = gpd.read_file(gpkg_file, layer=reservoir_layer)

    # Merge junctions and reservoirs for coordinate extraction
    all_nodes_gdf = gpd.GeoDataFrame(pd.concat([junctions_gdf, reservoirs_gdf], ignore_index=True))

    # Extract junctions and reservoirs
    junctions, reservoirs, coordinates, demands = extract_junctions_and_reservoirs(all_nodes_gdf)

    with open(inp_file, 'w') as file:
        file.write("[TITLE]\n")
        file.write("Generated from GeoPackage\n\n")

        file.write("[JUNCTIONS]\n")
        file.write(";ID     Elevation       Demand      Pattern\n")
        for junction in junctions:
            file.write(f"{junction}     {coordinates[junction][2]}       {demands[junction]} \n")
        file.write("\n")

        file.write("[RESERVOIRS]\n")
        file.write(";ID Head\n")
        for reservoir in reservoirs:
            file.write(f"{reservoir}        0 \n")
        file.write("\n")

        file.write("[PIPES]\n")
        file.write(";ID     Node1       Node2       Length      Diameter        Roughness       MinorLoss       Status\n")
        for _, row in pipes_gdf.iterrows():
            pipe_id = row['name']
            start = row['start']
            end = row['end']
            length = row['length']
            diameter = row.get('diameter', 200)  # Default to 200 if missing
            file.write(f"{pipe_id}      {start}     {end}       {length}        {diameter}      0.18     0       Open \n")
        file.write("\n")

        file.write("[COORDINATES]\n")
        file.write(";Node X-Coord Y-Coord\n")
        for key, value in coordinates.items():
            file.write(f"{key}      {value[0]}      {value[1]} \n")

    print(f"EPANET input file '{inp_file}' has been generated.")




#------ Example usage -------
gpkg_file = './Frodi_vatnsendi.gpkg'
junction_layer = "Junction"
pipe_layer = "Pipe"
reservoir_layer = "Reservoir"

inp_file = 'output_epanet_model.inp'
geopackage_to_epanet(gpkg_file, inp_file, junction_layer, pipe_layer, reservoir_layer)
