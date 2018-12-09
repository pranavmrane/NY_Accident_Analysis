import datetime
import geoplotlib as geoplotlib
import pandas as pd
import geocoder
from geoplotlib.utils import BoundingBox
from sklearn.cluster import KMeans
from uszipcode import SearchEngine
import matplotlib.pyplot as plt
import numpy as np
import collections
import operator
from collections import Counter

import warnings  #Add -Priya
warnings.filterwarnings("ignore", category=DeprecationWarning) #Priya


# Make an object of Search Engine which is a module of zipcode library
search = SearchEngine(simple_zipcode=True)

# Record KEY_VALUE
KEY_VALUE = 'As3n1BekOJNNi08upsGqp5omStD0C5GFd5EDFIhloqlXczFO1I0RKtmqCFqXDNaw'

# Boroughs with their zip codes are specified as global variables

BRONX = {10451, 10452, 10453, 10454, 10455, 10456, 10457, 10458, 10459,
         10460, 10461, 10462, 10463, 10464, 10465, 10466, 10467, 10468,
         10469, 10470, 10471, 10472, 10473, 10474, 10475}

BROOKLYN = {11201, 11203, 11204, 11205, 11206, 11207, 11208, 11209, 11210,
            11211, 11212, 11213, 11214, 11215, 11216, 11217, 11218, 11219,
            11220, 11221, 11222, 11223, 11224, 11225, 11226, 11228, 11229,
            11230, 11231, 11232, 11233, 11234, 11235, 11236, 11237, 11238,
            11239, 11241, 11242, 11243, 11249, 11252}

MANHATTAN = {10001, 10002, 10003, 10004, 10005, 10006, 10007, 10009, 10010,
             10011, 10012, 10013, 10014, 10015, 10016, 10017, 10018, 10019,
             10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028,
             10029, 10030, 10031, 10032, 10033, 10034, 10035, 10036, 10037,
             10038, 10039, 10040, 10041, 10044, 10045, 10048, 10055, 10060,
             10069, 10090, 10095, 10098, 10099, 10103, 10104, 10105, 10106,
             10107, 10110, 10111, 10112, 10115, 10118, 10119, 10120, 10121,
             10122, 10123, 10128, 10151, 10152, 10153, 10154, 10155, 10158,
             10161, 10162, 10165, 10166, 10167, 10168, 10169, 10170, 10171,
             10172, 10173, 10174, 10175, 10176, 10177, 10178, 10199, 10270,
             10271, 10278, 10279, 10280, 10281, 10282, 10065}

QUEENS = {11004, 11101, 11102, 11103, 11104, 11105, 11106, 11109, 11351,
          11354, 11355, 11356, 11357, 11358, 11359, 11360, 11361, 11362,
          11363, 11364, 11365, 11366, 11367, 11368, 11369, 11370, 11371,
          11372, 11373, 11374, 11375, 11377, 11378, 11379, 11385, 11411,
          11412, 11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420,
          11421, 11422, 11423, 11426, 11427, 11428, 11429, 11430, 11432,
          11433, 11434, 11435, 11436, 11691, 11692, 11693, 11694, 11697, 11005}

STATEN_ISLAND = {10301, 10302, 10303, 10304, 10305, 10306, 10307, 10308,
                 10309, 10310, 10311, 10312, 10314}

# There are accidents that take place outside the 5 boroughs
# The records with these Zip codes will be removed
NA = {1854, 7002, 7024, 7047, 7067, 7077, 7080, 7086, 7093, 7107, 7304, 7606,
      7632, 7645, 7646, 7656, 7718, 7747, 8244, 8401, 8840, 8853, 10504, 10512,
      10538, 10550, 10553, 10601, 10606, 10705, 10803, 10956, 10977, 11001,
      11003, 11042, 11050, 11096, 11509, 11516, 11520, 11542, 11552, 11556,
      11581, 11590, 11702, 11941, 12202, 12208, 12210, 12577, 12936, 13338,
      13346, 13409, 13601, 13787, 14530, 14607, 14853, 19716, 19966, 77078,
      98902}


def import_data(data_location):
    """
    Import Data as data frame
    :param data_location: Address relative to the project directory
    :return: data as dataframe
    """
    # Low memory is used to prevent warning while importing data
    base_dataset = pd.read_csv(data_location, low_memory=False)
    return base_dataset


def filter_data_by_date(base_dataset, start_range_date, end_range_date,
                        column_name):
    """
    Retain Data within a date range
    :param base_dataset: Dataset that will subsetted
    :param start_range_date: Start Date
    :param end_range_date: Last Date(No records from this date will be covered)
    :param column_name: Name of Date Column
    :return:
    """
    base_dataset[column_name] = pd.to_datetime(base_dataset[column_name])
    filtered_data = base_dataset[(base_dataset[column_name] > start_range_date)
                                 & (base_dataset[column_name] < end_range_date)
                                 ]
    # print(len(filtered_data.index))
    return filtered_data



def get_latitude_longitude(dataset):
    """
    Convert Address to GPS Co-ordinates.
    :param dataset: dataset that needs to be cleaned
    :return: dataset: Dataset that contains GPS Data for all rows
    """
    # print("get_latitude_longitude")

    # There are certain rows for which information cannot be found
    # These are recorded here
    drop_list = []

    for row_number, GPS_data in dataset['LOCATION'].iteritems():
        # If GPS data is absent or invalid
        if str(GPS_data) == "nan" or str(GPS_data) == '(0.0, 0.0)':

            correct_address = generate_street_address(
                dataset['ON STREET NAME'][row_number],
                dataset['CROSS STREET NAME'][row_number],
                dataset['OFF STREET NAME'][row_number])

            # Get Lat Long from street address
            get_location = geocoder.bing(correct_address, key=KEY_VALUE)

            # Sometimes for a street address, we do not get postal code
            if get_location.postal is None:
                # This means we did not get lat, long information as well
                if get_location.lat is None or get_location.lng is None:
                    # print("Value absolutely unusable " +
                    #       str(dataset['UNIQUE KEY'][row_number]))
                    drop_list.append(dataset['UNIQUE KEY'][row_number])
                    continue
                # Only postal code is None
                # We have lat, lon information for street address
                else:
                    # Attempting to find zip code for location
                    result = search.by_coordinates(get_location.lat,
                                                   get_location.lng, radius=30,
                                                   returns=1)
                    # We have found no results when we search
                    if len(result) == 0:
                        # print("LAZY PROBLEM")
                        drop_list.append(dataset['UNIQUE KEY'][row_number])
                        continue

                    verified_zip = result[0].zipcode
                    borough_prediction = get_borough_by_zip(verified_zip)
            else:
                # All expected data is present
                verified_zip = get_location.postal
                borough_prediction = get_borough_by_zip(verified_zip)

            if get_location != "None":
                try:
                    dataset.loc[row_number, 'LATITUDE'] = get_location.lat
                    dataset.loc[row_number, 'LONGITUDE'] = get_location.lng
                    dataset.loc[row_number, 'LOCATION'] = "(" + str(
                        get_location.lat) + "," + str(get_location.lng) + ")"

                    # ZIP CODE absent, BOROUGH present
                    # Add ZIP CODE
                    # Verify borough prediction
                    if (dataset.loc[row_number, 'ZIP CODE'] == 0 and
                            str(dataset.loc[row_number, 'BOROUGH']) != "nan"):
                        # Check if borough prediction matches the value
                        # originally present in dataset
                        if (str(dataset.loc[row_number, 'BOROUGH']) ==
                                borough_prediction):
                            dataset.loc[row_number, 'ZIP CODE'] = \
                                verified_zip
                        # There is a problem with prediction
                        # It is best if record is removed
                        else:
                            print("Borough Prediction does not match with "
                                  "already present borough-IF" +
                                  str(dataset['UNIQUE KEY'][row_number]) + " "
                                  + str(dataset.loc[row_number, 'BOROUGH']) +
                                  " " + borough_prediction)
                            drop_list.append(dataset['UNIQUE KEY'][row_number])

                    # ZIP CODE present, BOROUGH present
                    # Verify borough prediction
                    elif dataset.loc[row_number, 'ZIP CODE'] != 0 and str(
                            dataset.loc[row_number, 'BOROUGH']) != "nan":
                        if (str(dataset.loc[row_number, 'BOROUGH']) !=
                                borough_prediction):
                            # print("Borough Prediction does not match with "
                            #       "already present borough-ELIF " +
                            #       str(dataset['UNIQUE KEY'][row_number]) + " "
                            #       + str(dataset.loc[row_number, 'BOROUGH']) +
                            #       " " + borough_prediction)
                            drop_list.append(dataset['UNIQUE KEY'][row_number])

                    # ZIP CODE absent, BOROUGH absent
                    elif dataset.loc[row_number, 'ZIP CODE'] == 0 and str(
                            dataset.loc[row_number, 'BOROUGH']) == "nan":
                        dataset.loc[row_number, 'BOROUGH'] = borough_prediction
                        dataset.loc[row_number, 'ZIP CODE'] = verified_zip

                except Exception as e:
                    print(e)
                    print("get_latitude_longitude",
                          dataset['UNIQUE KEY'][row_number])
            else:
                print(
                    "get_location Not Found: " +
                    dataset['UNIQUE KEY'][row_number])

    print("Mismatched Borough will be dropped now")
    print(drop_list)

    if len(drop_list) > 0:
        # Remove rows by key
        dataset = remove_by_key(dataset, drop_list)

    # print("New Length", str(len(dataset.index)))
    return dataset


def generate_street_address(on_street, cross_street, off_street):
    """
    Generate Street Address based on the availability of Information
    :param on_street:
    :param cross_street:
    :param off_street:
    :return:
    """
    # Incase Off Street is absent
    if on_street != "" and cross_street != "":
        final_address = on_street + \
                        " " + "&" + " " + \
                        cross_street + \
                        " " + "NEW YORK CITY"
    elif on_street != "" and cross_street == "":
        # Generate address in format of 'On Street & Cross Street'
        final_address = on_street + " " + "NEW YORK CITY"
    else:
        # If Off Street information is present and
        # on-cross street information is absent
        final_address = off_street + \
                        "NEW YORK CITY"

    # Format String generated, generate proper spacing
    final_address = " ".join(final_address.split())

    return final_address


# def fill_all_empty_boroughs(dataset):
#     """
#
#     :param dataset:
#     :return:
#     """
#     print("fill_all_empty_boroughs")
#     print("Original Length", str(len(dataset.index)))
#     for row_number, borough_name in dataset['BOROUGH'].iteritems():
#         if str(dataset['BOROUGH'][row_number]) == 'nan' or
#             dataset['BOROUGH'][row_number] == 0):
#             dataset.loc[row_number, 'BOROUGH'] = \
#                 get_borough_by_zip(dataset['ZIP CODE'][row_number])
#
#     print("mid Length", str(len(dataset.index)))
#     # Remove values not in 5 boroughs
#     dataset = dataset.drop(dataset[(dataset.BOROUGH == "NA")].index)
#     print("New Length", str(len(dataset.index)))
#     return dataset


def add_borough_zip_by_lat_lon(dataset):
    """
    Find Zip by Latitude and Longitude
    Then Use Zip to get Borough
    :param dataset: Dataset to be updated, filtered
    :return: Updated, Filtered Dataset
    """
    # This will hold Keys of Rows with incorrect GPS Co-ordinates
    # Eg: Co-ordinates that point to middle of Atlantic Ocean
    drop_list = []

    for row_number, location_row in dataset['LOCATION'].iteritems():
        # Check all rows where borough field is empty OR
        # where ZIP code field is empty
        if (str(dataset['BOROUGH'][row_number]) == 'nan' or
                dataset['ZIP CODE'][row_number] == 0):

            # print("add_borough_zip_by_lat_lon " + str(row_number) + " " +
            #       str(dataset['UNIQUE KEY'][row_number]))

            # Get top result matching the latitude and longitude
            result = search.by_coordinates(dataset['LATITUDE'][row_number],
                                           dataset['LONGITUDE'][row_number],
                                           radius=30, returns=1)

            if len(result) == 0:
                # print("Odd GPS Value for Key: ",
                #       dataset['UNIQUE KEY'][row_number])
                # Save unique key where we got odd results
                # That row will be removed
                # Generally odd results point to middle of the ocean
                drop_list.append(dataset['UNIQUE KEY'][row_number])
            else:
                zip_code = result[0].zipcode
                try:
                    # Get borough from zip code
                    dataset.loc[row_number, 'BOROUGH'] = \
                        get_borough_by_zip(zip_code)
                    dataset.loc[row_number, 'ZIP CODE'] = zip_code
                except Exception as e:
                    print(e)
                    print("Some issue caught",
                          dataset['UNIQUE KEY'][row_number])

    print("Unique Keys for records with incorrect GPS co-ordinates will be "
          "removed: ")
    print(drop_list)

    if len(drop_list) > 0:
        # Remove rows by key
        dataset = remove_by_key(dataset, drop_list)

    # Remove values not in 5 boroughs
    dataset = dataset.drop(dataset[(dataset.BOROUGH == "NA")].index)

    return dataset


def replace_nan_in_some_columns(dataset):
    """
    There are nan values in dataset.
    Hence we replace nan with fixed keyword.
    :param dataset: dataset with incorrect text values
    :return: dataset:  fixed dataset
    """

    # Use 0 to replace nan
    dataset[['ZIP CODE']] = \
        dataset[['ZIP CODE']].fillna(value=0)
    # Convert Column to int
    dataset['ZIP CODE'] = dataset['ZIP CODE'].astype(int)
    # Use 0 to replace nan
    dataset[['ZIP CODE']] = \
        dataset[['ZIP CODE']].fillna(value=0)

    # Use 'Unspecified' to replace nan
    dataset[['CONTRIBUTING FACTOR VEHICLE 1',
             'CONTRIBUTING FACTOR VEHICLE 2',
             'CONTRIBUTING FACTOR VEHICLE 3',
             'CONTRIBUTING FACTOR VEHICLE 4',
             'CONTRIBUTING FACTOR VEHICLE 5']] = \
        dataset[['CONTRIBUTING FACTOR VEHICLE 1',
                 'CONTRIBUTING FACTOR VEHICLE 2',
                 'CONTRIBUTING FACTOR VEHICLE 3',
                 'CONTRIBUTING FACTOR VEHICLE 4',
                 'CONTRIBUTING FACTOR VEHICLE 5']].fillna(value='Unspecified')

    # Use 'UNKNOWN' to replace nan
    dataset[['VEHICLE TYPE CODE 1',
             'VEHICLE TYPE CODE 2',
             'VEHICLE TYPE CODE 3',
             'VEHICLE TYPE CODE 4',
             'VEHICLE TYPE CODE 5']] = \
        dataset[['VEHICLE TYPE CODE 1',
                 'VEHICLE TYPE CODE 2',
                 'VEHICLE TYPE CODE 3',
                 'VEHICLE TYPE CODE 4',
                 'VEHICLE TYPE CODE 5']].fillna(value='UNKNOWN')

    # Use '' to replace nan
    dataset[['ON STREET NAME',
             'CROSS STREET NAME',
             'OFF STREET NAME']] = \
        dataset[['ON STREET NAME',
                 'CROSS STREET NAME',
                 'OFF STREET NAME']].fillna(value='')

    return dataset


def remove_by_key(dataset, remove_list):
    """
    Remove rows from dataset that equal certain key. This is done to remove
    records with odd GPS co-ordinates.
    :param dataset: dataset to be cleaned
    :param remove_list: unique keys to be removed
    :return: cleaned dataset
    """

    for key in remove_list:
        dataset = dataset.drop(dataset[(dataset['UNIQUE KEY'] == key)].index)

    return dataset


def get_borough_by_zip(zipcode):
    """
    Return Borough by ZIP code
    :param zipcode: Zipcode
    :return: Borough name as String
    """
    if int(zipcode) in BRONX:
        return "BRONX"
    elif int(zipcode) in BROOKLYN:
        return "BROOKLYN"
    elif int(zipcode) in MANHATTAN:
        return "MANHATTAN"
    elif int(zipcode) in QUEENS:
        return "QUEENS"
    elif int(zipcode) in STATEN_ISLAND:
        return "STATEN ISLAND"
    elif int(zipcode) in NA:
        return "NA"
    else:
        return "NA"


def remove_incorrect_gps_records(dataset):
    """
    Ensure GPS Co-ordinates are in correct range. Drop the rows with wrong GPS
    values.
    :param dataset: Dataset with incorrect GPS Values
    :return: dataset: Dataset with mathematically correct GPS values
    """

    # print("remove_incorrect_gps_records")

    dataset = dataset.drop(dataset[(dataset.LONGITUDE > 180)].index)
    dataset = dataset.drop(dataset[(dataset.LONGITUDE < -180)].index)
    dataset = dataset.drop(dataset[(dataset.LATITUDE > 90)].index)
    dataset = dataset.drop(dataset[(dataset.LONGITUDE < -90)].index)

    return dataset


def remove_unusable_records(dataset):
    """
    Rows that contain GPS information, ZIP Code or address cannot be used
    They are being removed here
    :param dataset:
    :return:
    """

    drop_list = []
    # print("remove_unusable_records")

    for row_number, GPS_data in dataset['LOCATION'].iteritems():
        # If GPS data is absent or invalid
        if str(GPS_data) == "nan":
            # GPS Data, Borough Information and Street Address are missing
            # This type of record is unusable for this project
            if (dataset['OFF STREET NAME'][row_number] == "" and
                    dataset['ON STREET NAME'][row_number] == "" and
                    dataset['CROSS STREET NAME'][row_number] == ""):
                # Record Key to be removed
                drop_list.append(dataset['UNIQUE KEY'][row_number])

    print("Rows that contain no information are being listed. "
          "These will be dropped: ")
    print(drop_list)

    if len(drop_list) > 0:
        # Remove rows by key
        dataset = remove_by_key(dataset, drop_list)

    return dataset


def clean_time_filtered_dataset(dataset):
    """
    Clean time filtered dataset in the following way
    * Remove Mathematically incorrect GPS Values
    * Handle nan in Text Columns
    * Ensure all rows have Latitude and Longitude
    * Add Borough, Zip Code to data using GPS Co-ordinates
    :param dataset: Time Filtered Dataset
    :return: Fully Cleaned Dataset
    """
    # Remove Mathematically incorrect GPS Values
    filtered_data_gps_cleaned = remove_incorrect_gps_records(dataset)
    # Handle nan in Text Columns
    filtered_data_cleaned_replaced = replace_nan_in_some_columns(
        filtered_data_gps_cleaned)
    # Remove records that can't be used for this project
    filtered_data_cleaned_replaced_usable = remove_unusable_records(
        filtered_data_cleaned_replaced)
    # Ensure all rows have Latitude and Longitude
    filtered_data_lat_lon_added = get_latitude_longitude(
        filtered_data_cleaned_replaced_usable)
    # Add Borough, Zip Code to data using GPS Co-ordinates
    filtered_data_borough_added = add_borough_zip_by_lat_lon(
        filtered_data_lat_lon_added)
    # print(filtered_data_borough_added.isna().any())
    return filtered_data_borough_added



def factor_vehicle_accidents_graph(df_june_2017, df_july_2017,
                                   df_june_2018,df_july_2018):
    plt.title("Vehicle Accidents Occurance Distribution (Graph 1)")
    df = pd.DataFrame({'x': range(1, 54), 'Brooklyn_June_2017': df_june_2017,

                       'Brooklyn_July_2017':df_july_2017,'Brooklyn_June_2018':df_june_2018,
                       'Brooklyn_July_2018': df_july_2018})

    plt.plot('x','Brooklyn_June_2017',  data= df,  color='blue' )
    plt.plot('x', 'Brooklyn_July_2017', data= df,  color='red')
    plt.plot('x','Brooklyn_June_2018',  data= df,  color='green' )
    plt.plot('x', 'Brooklyn_July_2018', data= df,  color='yellow')
    plt.xticks(range(1,54), df.index, rotation='vertical')
    plt.legend()
    plt.show()

def visualize_heatmap_for_data(df):
    geo_data_for_plotting = {"lat": df["LATITUDE"],
                             "lon": df["LONGITUDE"]}
    geoplotlib.kde(geo_data_for_plotting, 1)
    east = max(df["LONGITUDE"])
    west = min(df["LONGITUDE"])
    south = max(df["LATITUDE"])
    north = min(df["LATITUDE"])
    bbox = BoundingBox(north=north, west=west, south=south, east=east)
    geoplotlib.set_bbox(bbox)
    geoplotlib.tiles_provider('toner-lite')
    geoplotlib.show()

def perform_clustering_kMeans(df, title):
    # filtering the dataset and save the coordinates
    data = df.filter(items=['LATITUDE', 'LONGITUDE'])
    # turn into a numpy array
    kmeans_array = np.array(data)

    # calculating the coordinates of the centroids of the clusters for 2,3 and 4 clusters
    for i in range(2, 5):
        kmeans2 = KMeans(n_clusters=i, random_state=0).fit(data)
        print(kmeans2.cluster_centers_)

    # centers = kmeans2.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


    #Lets choose k from 2 ,3,4 clusters to visualize results for data distribution
    for i in range(2, 5):
        results = KMeans(n_clusters=i,random_state=0).fit_predict(kmeans_array)
        plt.scatter(kmeans_array[:, 1],kmeans_array[:, 0], c=results, cmap='viridis')
        plt.title(title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()


def plot_hourly_distribution_graph(time, title):
    plt.hist(time, bins=50)
    plt.xlabel("Hours of the accident")
    plt.ylabel("Accident Count")
    plt.title(title)
    plt.show()

def plot_weekly_distribution_graph(week, title):
    plt.title(title)
    plt.xlabel("WeekDays and Weekends")
    plt.ylabel("Accident Count")
    plt.bar(range(len(week.keys())), week.values())
    weeks = ['Mon', 'Tues', 'Wedn', 'Thurs', 'Fri', 'Sat', 'Sun']
    plt.xticks(range(len(week.keys())), weeks)
    plt.show()

def less_and_high_accident_occurrences_with_dates(borough_accidents, year):
    borough_accident_ = collections.Counter(borough_accidents)
    sorted_by_num_of_accidents = \
        sorted(no_of_accidents_dates.items(), key=operator.itemgetter(1))

    print("Less Accidents in ", year, ":")
    less_Accidents = sorted_by_num_of_accidents[0:20]

    for date_ in less_Accidents:
        print(date_[0].strftime("%Y %b %d"), ":", date_[1])

    print("High Accidents in",year,":")
    high_Accidents = sorted_by_num_of_accidents[-20:]

    for date_ in high_Accidents:
        print(date_[0].strftime("%Y %b %d"), ":", date_[1])

    #View Accident count distributions across various boroughs of NYC
    plt.bar(range(5),
            borough_accident_.values())
    plt.xticks(range(5)
               , borough_accident_.keys())
    plt.xlabel("Borough Names")
    plt.ylabel("Accidents Count")
    # TODO: fix plot title - showing error
    # plt.gcf().set_size_inches(11,11)
    # plt.title("Accident Distributions for Boroughs for June-July ", year)
    plt.show()


if __name__ == '__main__':
    # original_dataset = import_data(
    #     "Data/NYPD_Motor_Vehicle_Collisions.csv"))
    # date1 = '2018-6-1 00:00:00'
    # date2 = '2018-8-1 00:00:00'
    # date3 = '2017-6-1 00:00:00'
    # date4 = '2017-8-1 00:00:00'
    # filtered_data_2018 = filter_data_by_date(original_dataset, date1, date2,
    #                                          'DATE')
    # filtered_data_2017 = filter_data_by_date(original_dataset, date3, date4,
    #                                          'DATE')

    # filtered_data_2018.to_csv("Data/Data_Filtered_By_Date_2018.csv", sep=',',
    #                      encoding='utf-8', index=False)
    # filtered_data_2017.to_csv("Data/Data_Filtered_By_Date_2017.csv", sep=',',
    #                           encoding='utf-8', index=False)

    # Presently we are directly using time filtered datasets to save time
    # print("Cleaning 2017 dataset:")
    # filtered_data_2017 = import_data(
    #     "Data/Data_Filtered_By_Date_2017.csv")
    # # filtered_data_2017 = filtered_data_2017.head(10000)
    #
    # fully_cleaned = clean_time_filtered_dataset(filtered_data_2017)
    # fully_cleaned.to_csv("Data/Data_Filtered_By_Date_2017_cleaned.csv",
    #                      sep=',', encoding='utf-8', index=False)
    #
    # print()
    # print("Cleaning 2018 dataset:")
    # filtered_data_2018 = import_data(
    #     "Data/Data_Filtered_By_Date_2018.csv")
    # fully_cleaned = clean_time_filtered_dataset(filtered_data_2018)
    # fully_cleaned = clean_time_filtered_dataset(filtered_data_2018)
    # fully_cleaned.to_csv("Data/Data_Filtered_By_Date_2018_cleaned.csv",
    #                      sep=',', encoding='utf-8', index=False)

    print("Phase 2: Visualization ")
    #Creating Dataframes for 2017 anad 2018
    df_2017 = import_data("Data/Data_Filtered_By_Date_2017_cleaned.csv")
    df_2018 = import_data("Data/Data_Filtered_By_Date_2018_cleaned.csv")
    df_2017['DATETIME'] = pd.to_datetime(df_2017['DATE'] + ' ' + df_2017['TIME'])
    # print(df_2017['CONTRIBUTING FACTOR VEHICLE 1'])

    #Printing no of rows and columns
    print(df_2018.shape)
    print(df_2017.shape)

    all_accident_borough_2017 = []
    borough_accident_2017= []
    no_of_accidents_dates = {}
    time_hour_accident= []
    accidents_weekdays = {}

    #Parsing and retrieving from 2017 Dataset
    for index, row in df_2017.iterrows():
        date = row['DATE']
        extracted_date = datetime.datetime.strptime(date,"%Y-%m-%d")
        weekly_date = datetime.datetime.strptime(date,"%Y-%m-%d").weekday()
        time_hours = row['TIME']

        try:
            no_of_accidents_dates[extracted_date] +=1
            accidents_weekdays[weekly_date] +=1
        except KeyError:
            no_of_accidents_dates[extracted_date] = 1
            accidents_weekdays[weekly_date] =1

        all_accident_borough_2017.append(row)
        borough_accident_2017.append(row['BOROUGH'])
        #Time
        if ':' in time_hours:
            time_hour = time_hours.split(":")
            time_hour_accident.append(int(time_hour[0]))

    #plotting hourly distribution of accidents  (2017)
    plot_hourly_distribution_graph(time_hour_accident,
                    "Hourly Distribution of Accidents in June-July 2017")


    #To sort and then count number of accidents weekly
    accidents_weekdays = sorted(accidents_weekdays.items())
    weekly_accidents_count = collections.OrderedDict(accidents_weekdays)
    #Plotting weekly accidents count
    plot_weekly_distribution_graph(weekly_accidents_count,
      "Weekly Distribution of June-July 2017" )


    # Lets try to see dates having high and low vehicles collisions (2017)
    # To count number of high accidents and low accidents
    less_and_high_accident_occurrences_with_dates(borough_accident_2017, "2017")



    # Parsing and retrieving from 2018 Dataset
    all_accident_borough_2018 = []
    borough_accident_2018= []
    for index, row in df_2018.iterrows():
        all_accident_borough_2018.append(row)
        borough_accident_2018.append(row['BOROUGH'])
        time_hours = row['TIME']
        # Time
        if ':' in time_hours:
            time_hour = time_hours.split(":")
            time_hour_accident.append(int(time_hour[0]))


    #plotting hourly distribution of accidents  (2018)
    plot_hourly_distribution_graph(time_hour_accident,
                                   "Hourly Distribution of Accidents in June-July 2018")

    #Plotting weekly distribution of accidents (2018):
    plot_weekly_distribution_graph(weekly_accidents_count,
                                   "Weekly Distribution of June-July 2018")

    #View Less and High Accident Occurances with dates in 2018:
    less_and_high_accident_occurrences_with_dates(borough_accident_2018,
                                                  "2018")


    #We decided to choose borough - 'Brooklyn'
    df_2017 = df_2017[df_2017['BOROUGH'] == 'BROOKLYN']
    df_2018 = df_2018[df_2018['BOROUGH'] == 'BROOKLYN']





    #Lets split clusters into June 2017, July 2017 , June 2018 and July 2018
    print("Phase 3: Clustering by use of kMeans:")
    df_june_2017 = filter_data_by_date(df_2017, pd.Timestamp(2017,6,1),
                                       pd.Timestamp(2017,6,30), 'DATE')
    df_july_2017 = filter_data_by_date(df_2017, pd.Timestamp(2017,7,1)
                           , pd.Timestamp(2017,7,31), 'DATE')

    df_june_2018 = filter_data_by_date(df_2018, pd.Timestamp(2018,6,1),
                                       pd.Timestamp(2018,6,30),'DATE')

    df_july_2018 = filter_data_by_date(df_2018, pd.Timestamp(2018,7,1),
                                       pd.Timestamp(2018,7,31),'DATE')



    vehicle_factor_count_june_2017 = df_june_2017.loc[(df_june_2017['CONTRIBUTING FACTOR VEHICLE 1'] != 'Unspecified')]
    vehicle_factor_count_june_2017 = vehicle_factor_count_june_2017['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()

    vehicle_factor_count_july_2017 = df_july_2017.loc[(df_july_2017['CONTRIBUTING FACTOR VEHICLE 1'] != 'Unspecified')]
    vehicle_factor_count_july_2017 = vehicle_factor_count_july_2017['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()

    vehicle_factor_count_june_2018 = df_june_2018.loc[(df_june_2018['CONTRIBUTING FACTOR VEHICLE 1'] != 'Unspecified')]
    vehicle_factor_count_june_2018 = vehicle_factor_count_june_2018['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()

    vehicle_factor_count_july_2018 = df_july_2018.loc[(df_july_2018['CONTRIBUTING FACTOR VEHICLE 1'] != 'Unspecified')]
    vehicle_factor_count_july_2018 = vehicle_factor_count_july_2018['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()



    #showing graph of viehicles accidents of 2017 and 2018
    factor_vehicle_accidents_graph(vehicle_factor_count_june_2017,vehicle_factor_count_july_2017,
                                   vehicle_factor_count_june_2018,
                                   vehicle_factor_count_july_2018)


    # Visualize heatmap for  june 2017
    visualize_heatmap_for_data(df_june_2017)


    # Visualize heatmap for  july 2017
    visualize_heatmap_for_data(df_june_2017)


    # #Similarily for june 2018
    visualize_heatmap_for_data(df_june_2018)

    # #Similarily for june 2018
    visualize_heatmap_for_data(df_july_2018)

    #Clustering of June 2017 , july 2018, june 2018,july 2018
    perform_clustering_kMeans(df_june_2017, "June 2017")

    # Clustering 2017 july
    perform_clustering_kMeans(df_july_2017, "July 2017")

    #Clustering June 2018
    perform_clustering_kMeans(df_june_2018, "June 2018")

    #July 2018
    perform_clustering_kMeans(df_july_2018, "July 2018")




"""
Output:
/home/pranavmrane/PycharmProjects/BDA_Project2/venv/bin/python /home/pranavmrane/PycharmProjects/BDA_Project2/Code/Data_Import_And_Clean.py
Phase 2: Visualization 
(39323, 29)
(39642, 30)
Less Accidents in  2017 :
2017 Jul 04 : 421
2017 Jun 04 : 462
2017 Jul 09 : 500
2017 Jul 29 : 501
2017 Jul 16 : 508
2017 Jul 02 : 528
2017 Jul 23 : 542
2017 Jun 25 : 576
2017 Jul 08 : 578
2017 Jul 01 : 582
2017 Jun 18 : 582
2017 Jul 25 : 587
2017 Jul 27 : 597
2017 Jul 30 : 601
2017 Jul 06 : 602
2017 Jul 03 : 609
2017 Jul 24 : 615
2017 Jul 11 : 623
2017 Jun 03 : 624
2017 Jul 17 : 629
High Accidents in 2017 :
2017 Jul 21 : 721
2017 Jun 02 : 722
2017 Jun 28 : 723
2017 Jun 30 : 732
2017 Jun 21 : 732
2017 Jul 19 : 733
2017 Jun 27 : 740
2017 Jun 14 : 741
2017 Jun 24 : 742
2017 Jul 13 : 746
2017 Jun 15 : 751
2017 Jun 29 : 754
2017 Jul 20 : 757
2017 Jun 09 : 758
2017 Jun 23 : 778
2017 Jun 12 : 795
2017 Jun 20 : 800
2017 Jun 19 : 802
2017 Jun 13 : 815
2017 Jun 22 : 833
Less Accidents in  2018 :
2017 Jul 04 : 421
2017 Jun 04 : 462
2017 Jul 09 : 500
2017 Jul 29 : 501
2017 Jul 16 : 508
2017 Jul 02 : 528
2017 Jul 23 : 542
2017 Jun 25 : 576
2017 Jul 08 : 578
2017 Jul 01 : 582
2017 Jun 18 : 582
2017 Jul 25 : 587
2017 Jul 27 : 597
2017 Jul 30 : 601
2017 Jul 06 : 602
2017 Jul 03 : 609
2017 Jul 24 : 615
2017 Jul 11 : 623
2017 Jun 03 : 624
2017 Jul 17 : 629
High Accidents in 2018 :
2017 Jul 21 : 721
2017 Jun 02 : 722
2017 Jun 28 : 723
2017 Jun 30 : 732
2017 Jun 21 : 732
2017 Jul 19 : 733
2017 Jun 27 : 740
2017 Jun 14 : 741
2017 Jun 24 : 742
2017 Jul 13 : 746
2017 Jun 15 : 751
2017 Jun 29 : 754
2017 Jul 20 : 757
2017 Jun 09 : 758
2017 Jun 23 : 778
2017 Jun 12 : 795
2017 Jun 20 : 800
2017 Jun 19 : 802
2017 Jun 13 : 815
2017 Jun 22 : 833
Phase 3: Clustering by use of kMeans:
('smallest non-zero count', 8.577404340105357e-05)
('max count:', 2.516574822254173)
('smallest non-zero count', 8.577404340105357e-05)
('max count:', 2.516574822254173)
('smallest non-zero count', 1.791063608477192e-08)
('max count:', 6.047982810383686)
('smallest non-zero count', 8.577404340105357e-05)
('max count:', 6.843781910490505)
('smallest non-zero count', 3.582127216954384e-08)
('max count:', 43.60205395967483)
('smallest non-zero count', 1.791063608477192e-08)
('max count:', 18.92057927333186)
('smallest non-zero count', 3.582127216954384e-08)
('max count:', 43.60205395967483)
('smallest non-zero count', 5.373190825431576e-08)
('max count:', 7.777359860347349)
('smallest non-zero count', 1.791063608477192e-08)
('max count:', 7.107970808541595)
('smallest non-zero count', 1.8330883346989723e-06)
('max count:', 7.077911292619715)
('smallest non-zero count', 0.00016447682499448482)
('max count:', 6.855363512261463)
('smallest non-zero count', 1.791063608477192e-08)
('max count:', 7.107970808541595)
('smallest non-zero count', 5.373190825431576e-08)
('max count:', 7.777359860347349)
[[ 40.62338193 -73.98121531]
 [ 40.67842194 -73.93430536]]
[[ 40.69067037 -73.96508691]
 [ 40.66052491 -73.90787237]
 [ 40.61728946 -73.98142487]]
[[ 40.66562048 -73.90546997]
 [ 40.61095811 -73.95305745]
 [ 40.63008976 -74.00336659]
 [ 40.69183951 -73.96371876]]
[[ 40.67804831 -73.93254022]
 [ 40.62459805 -73.98217559]]
[[ 40.6902793  -73.96406408]
 [ 40.61750545 -73.98306641]
 [ 40.66002943 -73.90659069]]
[[ 40.6634265  -73.90392624]
 [ 40.60832767 -73.95986801]
 [ 40.69191576 -73.95936781]
 [ 40.64071029 -74.00567032]]
[[ 40.62231355 -73.98107488]
 [ 40.67813648 -73.93411387]]
[[ 40.61666255 -73.98331582]
 [ 40.68979571 -73.96412917]
 [ 40.65909844 -73.90789732]]
[[ 40.63157607 -74.00369134]
 [ 40.66500645 -73.90551637]
 [ 40.60803387 -73.95515504]
 [ 40.69093913 -73.96288597]]
[[ 40.67718478 -73.93212761]
 [ 40.62392539 -73.98201909]]
[[ 40.6895689  -73.96368245]
 [ 40.61678314 -73.98427534]
 [ 40.65831731 -73.90736883]]
[[ 40.60845821 -73.95446848]
 [ 40.69032862 -73.96185279]
 [ 40.66429752 -73.90397111]
 [ 40.63249645 -74.00547211]]

Process finished with exit code 0


"""
