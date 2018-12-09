import pandas as pd
import geocoder
from uszipcode import SearchEngine


class DataCleaner:

    def __init__(self):
        pass

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
              11433, 11434, 11435, 11436, 11691, 11692, 11693, 11694, 11697,
              11005}

    STATEN_ISLAND = {10301, 10302, 10303, 10304, 10305, 10306, 10307, 10308,
                     10309, 10310, 10311, 10312, 10314}

    # There are accidents that take place outside the 5 boroughs
    # The records with these Zip codes will be removed
    NA = {1854, 7002, 7024, 7047, 7067, 7077, 7080, 7086, 7093, 7107, 7304,
          7606,
          7632, 7645, 7646, 7656, 7718, 7747, 8244, 8401, 8840, 8853, 10504,
          10512,
          10538, 10550, 10553, 10601, 10606, 10705, 10803, 10956, 10977, 11001,
          11003, 11042, 11050, 11096, 11509, 11516, 11520, 11542, 11552, 11556,
          11581, 11590, 11702, 11941, 12202, 12208, 12210, 12577, 12936, 13338,
          13346, 13409, 13601, 13787, 14530, 14607, 14853, 19716, 19966, 77078,
          98902}

    def import_data(self, data_location):
        """
        Import Data as data frame
        :param data_location: Address relative to the project directory
        :return: data as dataframe
        """
        # Low memory is used to prevent warning while importing data
        base_dataset = pd.read_csv(data_location, low_memory=False)
        return base_dataset

    def filter_data_by_date(self, base_dataset, start_range_date,
                            end_range_date,
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
        filtered_data = base_dataset[
            (base_dataset[column_name] > start_range_date)
            & (base_dataset[column_name] < end_range_date)
            ]
        # print(len(filtered_data.index))
        return filtered_data

    def get_latitude_longitude(self, dataset):
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

                correct_address = self.generate_street_address(
                    dataset['ON STREET NAME'][row_number],
                    dataset['CROSS STREET NAME'][row_number],
                    dataset['OFF STREET NAME'][row_number])

                # Get Lat Long from street address
                get_location = geocoder.bing(correct_address,
                                             key=self.KEY_VALUE)

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
                        result = self.search.by_coordinates(get_location.lat,
                                                            get_location.lng,
                                                            radius=30,
                                                            returns=1)
                        # We have found no results when we search
                        if len(result) == 0:
                            # print("LAZY PROBLEM")
                            drop_list.append(dataset['UNIQUE KEY'][row_number])
                            continue

                        verified_zip = result[0].zipcode
                        borough_prediction = self.get_borough_by_zip(
                            verified_zip)
                else:
                    # All expected data is present
                    verified_zip = get_location.postal
                    borough_prediction = self.get_borough_by_zip(verified_zip)

                if get_location != "None":
                    try:
                        dataset.loc[row_number, 'LATITUDE'] = get_location.lat
                        dataset.loc[row_number, 'LONGITUDE'] = get_location.lng
                        dataset.loc[row_number, 'LOCATION'] = "(" + str(
                            get_location.lat) + "," + str(
                            get_location.lng) + ")"

                        # ZIP CODE absent, BOROUGH present
                        # Add ZIP CODE
                        # Verify borough prediction
                        if (dataset.loc[row_number, 'ZIP CODE'] == 0 and
                                str(dataset.loc[
                                        row_number, 'BOROUGH']) != "nan"):
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
                                      str(dataset['UNIQUE KEY'][
                                              row_number]) + " "
                                      + str(
                                    dataset.loc[row_number, 'BOROUGH']) +
                                      " " + borough_prediction)
                                drop_list.append(
                                    dataset['UNIQUE KEY'][row_number])

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
                                drop_list.append(
                                    dataset['UNIQUE KEY'][row_number])

                        # ZIP CODE absent, BOROUGH absent
                        elif dataset.loc[row_number, 'ZIP CODE'] == 0 and str(
                                dataset.loc[row_number, 'BOROUGH']) == "nan":
                            dataset.loc[
                                row_number, 'BOROUGH'] = borough_prediction
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
            dataset = self.remove_by_key(dataset, drop_list)

        # print("New Length", str(len(dataset.index)))
        return dataset

    def generate_street_address(self, on_street, cross_street, off_street):
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

    def add_borough_zip_by_lat_lon(self, dataset):
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
                result = self.search.by_coordinates(
                    dataset['LATITUDE'][row_number],
                    dataset['LONGITUDE'][
                        row_number],
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
                            self.get_borough_by_zip(zip_code)
                        dataset.loc[row_number, 'ZIP CODE'] = zip_code
                    except Exception as e:
                        print(e)
                        print("Some issue caught",
                              dataset['UNIQUE KEY'][row_number])

        print(
            "Unique Keys for records with incorrect GPS co-ordinates will be "
            "removed: ")
        print(drop_list)

        if len(drop_list) > 0:
            # Remove rows by key
            dataset = self.remove_by_key(dataset, drop_list)

        # Remove values not in 5 boroughs
        dataset = dataset.drop(dataset[(dataset.BOROUGH == "NA")].index)

        return dataset

    def replace_nan_in_some_columns(self, dataset):
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
                     'CONTRIBUTING FACTOR VEHICLE 5']].fillna(
                value='Unspecified')

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

    def remove_by_key(self, dataset, remove_list):
        """
        Remove rows from dataset that equal certain key. This is done to remove
        records with odd GPS co-ordinates.
        :param dataset: dataset to be cleaned
        :param remove_list: unique keys to be removed
        :return: cleaned dataset
        """

        for key in remove_list:
            dataset = dataset.drop(
                dataset[(dataset['UNIQUE KEY'] == key)].index)

        return dataset

    def get_borough_by_zip(self, zipcode):
        """
        Return Borough by ZIP code
        :param zipcode: Zipcode
        :return: Borough name as String
        """
        if int(zipcode) in self.BRONX:
            return "BRONX"
        elif int(zipcode) in self.BROOKLYN:
            return "BROOKLYN"
        elif int(zipcode) in self.MANHATTAN:
            return "MANHATTAN"
        elif int(zipcode) in self.QUEENS:
            return "QUEENS"
        elif int(zipcode) in self.STATEN_ISLAND:
            return "STATEN ISLAND"
        elif int(zipcode) in self.NA:
            return "NA"
        else:
            return "NA"

    def remove_incorrect_gps_records(self, dataset):
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

    def remove_unusable_records(self, dataset):
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
            dataset = self.remove_by_key(dataset, drop_list)

        return dataset

    def clean_time_filtered_dataset(self, dataset):
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
        filtered_data_gps_cleaned = self.remove_incorrect_gps_records(dataset)
        # Handle nan in Text Columns
        filtered_data_cleaned_replaced = self.replace_nan_in_some_columns(
            filtered_data_gps_cleaned)
        # Remove records that can't be used for this project
        filtered_data_cleaned_replaced_usable = self.remove_unusable_records(
            filtered_data_cleaned_replaced)
        # Ensure all rows have Latitude and Longitude
        filtered_data_lat_lon_added = self.get_latitude_longitude(
            filtered_data_cleaned_replaced_usable)
        # Add Borough, Zip Code to data using GPS Co-ordinates
        filtered_data_borough_added = self.add_borough_zip_by_lat_lon(
            filtered_data_lat_lon_added)
        # print(filtered_data_borough_added.isna().any())
        return filtered_data_borough_added
