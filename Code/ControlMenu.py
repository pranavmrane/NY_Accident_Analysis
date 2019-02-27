"""
CSCI720 Project 2
NAme: ControlMenu.py (Main)
Program to evaluate , provide evidence of learning and
understanding of assignment on NYC Traffic Collisions
Date: 1st Dec 2018
Authors:Priyanka Patil (pxp1439@rit.edu)
Authors: Pranav Rane (pmr5279@rit.edu)
"""
from Code.DataCluster import DataCluster
from Code.DataVisualizer import DataVisualizer
from Code.DataCleaner import DataCleaner
import csv

import pandas as pd

if __name__ == '__main__':
    import_object = DataCleaner()

    date1 = '2018-6-1 00:00:00'
    date2 = '2018-8-1 00:00:00'
    date3 = '2017-6-1 00:00:00'
    date4 = '2017-8-1 00:00:00'

    # original_dataset = import_object. \
    #     import_data(
    #     "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD")


    # filtered_data_2017 = import_object.filter_data_by_date(original_dataset,
    #                                                        date3, date4,
    #                                                        'DATE')
    #
    # filtered_data_2018 = import_object.filter_data_by_date(original_dataset,
    #                                                        date1, date2,
    #                                                        'DATE')

    data_2017 = '../Data/Data_Filtered_By_Date_2017.csv'
    data_2017_file = pd.read_csv(data_2017)

    data_2018 = '../Data/Data_Filtered_By_Date_2018.csv'
    data_2018_file = pd.read_csv(data_2018)

    filtered_data_2017 = import_object.filter_data_by_date(data_2017_file,
                                                           date3, date4,
                                                           'DATE')

    filtered_data_2018 = import_object.filter_data_by_date(data_2018_file,
                                                           date1, date2,
                                                           'DATE')


    df_2017 = import_object.clean_time_filtered_dataset(
        filtered_data_2017)

    df_2018 = import_object.clean_time_filtered_dataset(
        filtered_data_2018)

    visualizer_object = DataVisualizer()


    df_2017 = visualizer_object.make_date_time(df_2017)
    df_2018 = visualizer_object.make_date_time(df_2018)

    df_2017, df_2018 = visualizer_object.high_processing(df_2017, df_2018)

    # Lets split clusters into June 2017, July 2017 , June 2018 and July 2018
    print("Clustering by use of kMeans:")
    clustering_object = DataCluster()
    df_june_2017 = import_object.filter_data_by_date(df_2017,
                                                     pd.Timestamp(2017, 6, 1),
                                                     pd.Timestamp(2017, 7, 1),
                                                     'DATE')
    df_july_2017 = import_object.filter_data_by_date(df_2017,
                                                     pd.Timestamp(2017, 7, 2)
                                                     ,
                                                     pd.Timestamp(2017, 7, 31),
                                                     'DATE')

    df_june_2018 = import_object.filter_data_by_date(df_2018,
                                                     pd.Timestamp(2018, 6, 1),
                                                     pd.Timestamp(2018, 6, 30),
                                                     'DATE')

    df_july_2018 = import_object.filter_data_by_date(df_2018,
                                                     pd.Timestamp(2018, 7, 1),
                                                     pd.Timestamp(2018, 7, 31),
                                                     'DATE')

    vehicle_factor_count_june_2017 = clustering_object.remove_unspecified(
        df_june_2017)
    vehicle_factor_count_july_2017 = clustering_object.remove_unspecified(
        df_july_2017)
    vehicle_factor_count_june_2018 = clustering_object.remove_unspecified(
        df_june_2018)
    vehicle_factor_count_july_2018 = clustering_object.remove_unspecified(
        df_july_2018)

    # showing graph of viehicles accidents of 2017 and 2018
    clustering_object.factor_vehicle_accidents_graph(
        vehicle_factor_count_june_2017,
        vehicle_factor_count_july_2017,
        vehicle_factor_count_june_2018,
        vehicle_factor_count_july_2018)



    # View accident severity scores across NYC Boroughs:
    # Calculate severity score like low cost for injury and high cost for death penalty
    #For June and July 2017 :
    collisions_in_brooklyn_2017 = visualizer_object.severity_score_measure(df_2017)
    # Perform kmeans choosing 15 centroids to find hotspots
    centroids, labels = visualizer_object.get_centroids_clusters(collisions_in_brooklyn_2017, 15)

    print(visualizer_object.map_collisions_visualize(collisions_in_brooklyn_2017,
                                   centroids=centroids,  # The centroids of collisions via kmeans
                                   labels=labels,  # labels of clusters
                                   only_show_num_collisions=0,
                                   output_file_name="JuneJuly_2017"))


    # For June and July 2018
    # Calculate severity score like low cost for injury and high cost for death penalty
    collisions_in_brooklyn_2018 = visualizer_object.severity_score_measure(df_2018)
    # Run kmeans with 15 centroids (predefined)
    centroids, labels = visualizer_object.get_centroids_clusters(collisions_in_brooklyn_2018, 15)

    print(visualizer_object.map_collisions_visualize(collisions_in_brooklyn_2018,
                                   centroids=centroids,  # The centroids of collisions via through kmeans
                                   labels=labels,  # labels of respective collision clusters
                                   only_show_num_collisions=0,
                                   output_file_name="JuneJuly_2018"))


    #Visualize number of pedestrains killed/injured - for pedestrains safety
    # visualizer_object.view_number_of_pedestrains_killed(df_2017,df_2018)

    # Visualize different factors - pedestrians injured and killed for
    # june july 2017, 2018
    pedestrains_injured = visualizer_object.get_count_pedestrains(df_june_2017, 'NUMBER OF PEDESTRIANS INJURED', 'June 2017')

    pedestrains_killed = visualizer_object.get_count_pedestrains(df_june_2017, 'NUMBER OF PEDESTRIANS KILLED', 'June 2017')

    pedestrains_injured = visualizer_object.get_count_pedestrains(df_july_2017, 'NUMBER OF PEDESTRIANS INJURED',
                                                'July 2017')
    pedestrains_killed = visualizer_object.get_count_pedestrains(df_july_2017, 'NUMBER OF PEDESTRIANS KILLED', 'July 2017')

    pedestrains_injured = visualizer_object.get_count_pedestrains(df_june_2018, 'NUMBER OF PEDESTRIANS INJURED',
                                                'June 2018')
    pedestrains_killed = visualizer_object.get_count_pedestrains(df_june_2018, 'NUMBER OF PEDESTRIANS KILLED', 'June 2018')

    pedestrains_injured = visualizer_object.get_count_pedestrains(df_july_2018, 'NUMBER OF PEDESTRIANS INJURED',
                                                'July 2018')
    pedestrains_killed = visualizer_object.get_count_pedestrains(df_july_2018, 'NUMBER OF PEDESTRIANS KILLED', 'July 2018')

    # Need pedestrains count for combinations of June July 2017 vs
    # June July 2018
    pedestrains_injured = visualizer_object.get_count_pedestrains(df_2017, 'NUMBER OF PEDESTRIANS INJURED',
                                                                  'JuneJuly 2017')

    pedestrains_killed = visualizer_object.get_count_pedestrains(df_2017, 'NUMBER OF PEDESTRIANS KILLED',
                                                                 'JuneJuly 2017')
    pedestrains_injured = visualizer_object.get_count_pedestrains(df_2018, 'NUMBER OF PEDESTRIANS INJURED',
                                                                  'JuneJuly 2018')

    pedestrains_killed = visualizer_object.get_count_pedestrains(df_2018, 'NUMBER OF PEDESTRIANS KILLED',
                                                                 'JuneJuly 2018')


"""
Output:
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
2017 Jun 18 : 582
High Accidents in 2017 :
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
2017 Jun 18 : 582
High Accidents in 2018 :
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
file:///home/priyanka/PycharmProjects/BDA/NY_Accident_Analysis/Code/JuneJuly_2017.html
file:///home/priyanka/PycharmProjects/BDA/NY_Accident_Analysis/Code/JuneJuly_2018.html
254  collisions has  number of pedestrians injured in June 2017
1  collisions has  number of pedestrians killed in June 2017
199  collisions has  number of pedestrians injured in July 2017
3  collisions has  number of pedestrians killed in July 2017
243  collisions has  number of pedestrians injured in June 2018
2  collisions has  number of pedestrians killed in June 2018
217  collisions has  number of pedestrians injured in July 2018
3  collisions has  number of pedestrians killed in July 2018
482  collisions has  number of pedestrians injured in JuneJuly 2017
4  collisions has  number of pedestrians killed in JuneJuly 2017
479  collisions has  number of pedestrians injured in JuneJuly 2018
6  collisions has  number of pedestrians killed in JuneJuly 2018

Process finished with exit code 0

"""