from Code.DataCleaner import DataCleaner

if __name__ == '__main__':
    import_object = DataCleaner()

    original_dataset = import_object. \
        import_data(
        "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD")
    date1 = '2018-6-1 00:00:00'
    date2 = '2018-8-1 00:00:00'
    date3 = '2017-6-1 00:00:00'
    date4 = '2017-8-1 00:00:00'
    filtered_data_2018 = import_object.filter_data_by_date(original_dataset,
                                                           date1, date2,
                                                           'DATE')
    filtered_data_2017 = import_object.filter_data_by_date(original_dataset,
                                                           date3, date4,
                                                           'DATE')

    filtered_data_2018.to_csv("Data/Data_Filtered_By_Date_2018.csv", sep=',',
                              encoding='utf-8', index=False)
    filtered_data_2017.to_csv("Data/Data_Filtered_By_Date_2017.csv", sep=',',
                              encoding='utf-8', index=False)

    # Presently we are directly using time filtered datasets to save time
    print("Cleaning 2017 dataset:")
    filtered_data_2017 = import_object.import_data(
        "Data/Data_Filtered_By_Date_2017.csv")
    # filtered_data_2017 = filtered_data_2017.head(10000)

    fully_cleaned = import_object.clean_time_filtered_dataset(
        filtered_data_2017)
    fully_cleaned.to_csv("Data/Data_Filtered_By_Date_2017_cleaned.csv",
                         sep=',', encoding='utf-8', index=False)

    print()
    print("Cleaning 2018 dataset:")
    filtered_data_2018 = import_object.import_data(
        "Data/Data_Filtered_By_Date_2018.csv")
    fully_cleaned = import_object.clean_time_filtered_dataset(
        filtered_data_2018)
    fully_cleaned.to_csv("Data/Data_Filtered_By_Date_2018_cleaned.csv",
                         sep=',', encoding='utf-8', index=False)
