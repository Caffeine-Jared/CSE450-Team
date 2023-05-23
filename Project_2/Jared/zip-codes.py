import pgeocode
import pandas as pd
import requests

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')

# get unique zip codes from df
zip_codes = df['zipcode'].unique()

nomi = pgeocode.Nominatim('us')

# prepare empty lists for latitudes and longitudes
latitudes = []
longitudes = []

# loop through zip codes and get lat and long
for zip in zip_codes:
    location = nomi.query_postal_code(str(zip))
    latitudes.append(location['latitude'])
    longitudes.append(location['longitude'])

# create a dictionary with latitudes and longitudes
zip_dict = {'zipcode': zip_codes, 'lat_zip': latitudes, 'long_zip': longitudes}

#%%

# locations_hq = {
#     "Amazon_HQ": (47.62246, -122.336775),
#     "Microsoft": (47.64429, -122.12518),
#     "Starbucks": (47.580463, -122.335897),
#     "Boeing_Plant": (47.543969, -122.316443)
# }

# # Create a new DataFrame from the dictionary
# df_locations_hq = pd.DataFrame.from_dict(locations_hq, orient='index', columns=['Lat_hq', 'Long_hq'])

# print(df_locations_hq)

# print(zip_dict)
# create a new dataframe from the dictionary
df_zipcodes = pd.DataFrame(zip_dict)
print(df_zipcodes)


def get_distance(api_key, origin, destination):
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "key": api_key,
        "units": "imperial"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if "distance" in data["rows"][0]["elements"][0]:
        distance_text = data["rows"][0]["elements"][0]["distance"]["text"]
        # Extract the number from the returned string, convert it to float and then to miles
        distance = float(distance_text.split()[0])
    else:
        distance = None
    return distance

# Assume you already have pandas dataframe called df_zipcodes
# df_zipcodes = pd.read_csv('zipcodes_file.csv') # If you read data from a csv file

api_key = input("Enter your Google API key: ")

locations_hq = {
    "Amazon_HQ": (47.62246, -122.336775),
    "Microsoft": (47.64429, -122.12518),
    "Starbucks": (47.580463, -122.335897),
    "Boeing_Plant": (47.543969, -122.316443)
}

for name, (lat, long) in locations_hq.items():
    distances = []
    for _, row_zip in df_zipcodes.iterrows():
        origin = f"{row_zip['lat_zip']},{row_zip['long_zip']}"
        destination = f"{lat},{long}"
        distance = get_distance(api_key, origin, destination)
        distances.append(distance)
    df_zipcodes[name] = distances

df_zipcodes.to_csv('zipcodes_file.csv', index=False)
print(df_zipcodes)