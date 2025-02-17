import requests
from datetime import datetime

# Function to format dates into the required timestamp format
def format_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        print("Invalid date format, please use 'YYYY-MM-DD HH:MM:SS'.")
        return None

# Function to poll the database with custom date range
def poll_database(start_date, end_date, device_name):
    # Format the dates into the required format
    start_date = format_date(start_date)
    end_date = format_date(end_date)

    if not start_date or not end_date:
        return
    
    # Base URL
    url = "https://services2.arcgis.com/rmrALMmpPh0iZNsU/arcgis/rest/services/Layer_1_view/FeatureServer/0/query"
    
    # Construct the query parameters
    where_clause = f"(time_date BETWEEN timestamp '{start_date}' AND timestamp '{end_date}') AND (device_name='{device_name}')"
    
    params = {
        "f": "json",
        "cacheHint": "true",
        "outFields": "*",
        "outStatistics": '[{"onStatisticField":"time_date","outStatisticFieldName":"mindate","statisticType":"min"},{"onStatisticField":"time_date","outStatisticFieldName":"maxdate","statisticType":"max"}]',
        "resultType": "standard",
        "returnGeometry": "false",
        "spatialRel": "esriSpatialRelIntersects",
        "where": where_clause
    }

    # Send the GET request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        # Process the response as needed
        print("Data retrieved successfully:")
        print(data)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

# Example usage: Modify the date range and device name
start_date = "2025-02-09 23:00:00"
end_date = "2025-02-16 22:59:59"
device_name = "29_Samskipahöllin"

poll_database(start_date, end_date, device_name)
