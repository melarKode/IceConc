import cdsapi
import os
import zipfile

dataset = "satellite-sea-ice-concentration"
request = {
    "variable": "all",
    "version": "v3",
    "sensor": "amsr",
    "origin": "eumetsat_osi_saf",
    "region": ["northern_hemisphere"],
    "cdr_type": ["cdr"],
    "temporal_aggregation": "daily",
    "year": ["2002", "2003"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ]
}

if __name__ == "__main__":
    client = cdsapi.Client()

    if not os.path.exists("data"):
        os.makedirs("data")

    # # Creating pairs to avoid exceeding the 1000 request limit
    year_pairs = [[str(i), str(i + 1)] for i in range (2002, 2021, 2)]

    # print(f"Year pairs for downloading: {year_pairs}")

    # request_pairs = [request.copy() for _ in year_pairs]
    zip_file_path = [f"data/sea_ice_concentration_{year_pair[0]}_{year_pair[1]}.zip" for year_pair in year_pairs]
    # for i, year_pair in enumerate(year_pairs):
    #     request_pairs[i]["year"] = year_pair
    #     # Downloading data in pairs of years
    #     client.retrieve(dataset, request_pairs[i], zip_file_path[i])
    #     print(f"Downloaded data for years {year_pair[0]} and {year_pair[1]} to {zip_file_path[i]}")

    # Unzipping downloaded files
    for zip_path in zip_file_path:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            try:
                zip_ref.extractall("data")
                print(f"Extracted {zip_path} to data/")
            except zipfile.BadZipFile:
                print(f"Failed to extract {zip_path}. It may not be a valid zip file or is empty.")
            finally:
                os.remove(zip_path)
                print(f"Removed zip file {zip_path} after extraction.")
    print("All data downloaded and extracted successfully.")
    print("Data is ready for use in the 'data' directory.")



