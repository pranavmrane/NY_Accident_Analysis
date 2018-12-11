"""
CSCI720 Project 2
NAme: ControlMenu.py (Main)
Program to evaluate , provide evidence of learning and
understanding of assignment on NYC Traffic Collisions
Date: 1st Dec 2018
Authors:Priyanka Patil (pxp1439@rit.edu)
Authors: Pranav Rane (pmr5279@rit.edu)
"""
from Code.DataCleaner import DataCleaner
from Code.DataVisualizer import DataVisualizer
from Code.DataCluster import DataCluster
import pandas as pd


if __name__ == '__main__':
    import_object = DataCleaner()

    # original_dataset = import_object. \
    #     import_data(
    #     "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD")
    # date1 = '2018-6-1 00:00:00'
    # date2 = '2018-8-1 00:00:00'
    # date3 = '2017-6-1 00:00:00'
    # date4 = '2017-8-1 00:00:00'
    # filtered_data_2018 = import_object.filter_data_by_date(original_dataset,
    #                                                        date1, date2,
    #                                                        'DATE')
    # filtered_data_2017 = import_object.filter_data_by_date(original_dataset,
    #                                                        date3, date4,
    #                                                        'DATE')
    #
    # df_2017 = import_object.clean_time_filtered_dataset(
    #     filtered_data_2017)
    #
    # df_2018 = import_object.clean_time_filtered_dataset(
    #     filtered_data_2018)

    visualizer_object = DataVisualizer()

    # Creating Dataframes for 2017 anad 2018
    df_2017 = import_object.import_data(
        "../Data/Data_Filtered_By_Date_2017_cleaned.csv")
    df_2018 = import_object.import_data(
        "../Data/Data_Filtered_By_Date_2018_cleaned.csv")

    df_2017 = visualizer_object.make_date_time(df_2017)
    df_2018 = visualizer_object.make_date_time(df_2018)

    # Printing no of rows and columns
    # print(df_2017.shape)
    # print(df_2018.shape)

    df_2017, df_2018 = visualizer_object.high_processing(df_2017, df_2018)

    # Lets split clusters into June 2017, July 2017 , June 2018 and July 2018
    print("Phase 3: Clustering by use of kMeans:")
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

    #  Visualize heatmap for  june 2017
    clustering_object.visualize_heatmap_for_data(df_june_2017)

    #  Visualize heatmap for  july 2017
    clustering_object.visualize_heatmap_for_data(df_june_2017)

    # Similarily for june 2018
    clustering_object.visualize_heatmap_for_data(df_june_2018)

    # Similarily for june 2018
    clustering_object.visualize_heatmap_for_data(df_july_2018)

    # Clustering of June 2017 , july 2018, june 2018,july 2018
    clustering_object.perform_clustering_kMeans(df_june_2017, " Results for Clustering under June 2017")

    #Clustering 2017 july
    clustering_object.perform_clustering_kMeans(df_july_2017, " Results for Clustering under July 2017")

    # Clustering June 2018
    clustering_object.perform_clustering_kMeans(df_june_2018, "  Results for Clustering under June 2018")
    #
    # Similarily for July 2018
    clustering_object.perform_clustering_kMeans(df_july_2018, " Results for Clustering under July 2018")


    #View accident severity scores across NYC Boroughs:
    # Calculate severity score like low cost for injury and high cost for death penalty
    collisions_in_brooklyn_june_2017 = visualizer_object.severity_score_measure(df_june_2017)
    # Perform kmeans choosing 15 centroids to find hotspots
    centroids, labels = visualizer_object.get_centroids_clusters(collisions_in_brooklyn_june_2017, 15)

    print(visualizer_object.map_collisions_visualize(collisions_in_brooklyn_june_2017,  # The list of collisions to plot
                                   centroids=centroids,  # The centroids of collisions, chosen through kmeans
                                   labels=labels,  # The list of which collisions belong to which centroid
                                   only_show_num_collisions=0,
                                   # Randomly pick only some collisions to show, for a cleaner map
                                   output_file_name="june 2017"))

    # July 2017
    # Calculate severity score like low cost for injury and high cost for death penalty
    collisions_in_brooklyn_july_2017 = visualizer_object.severity_score_measure(df_july_2017)
    # Run kmeans with 15 centroids, find 15 'collision hotspots'
    centroids, labels = visualizer_object.get_centroids_clusters(collisions_in_brooklyn_july_2017, 15)

    print(visualizer_object.map_collisions_visualize(collisions_in_brooklyn_july_2017,  # The list of collisions to plot
                                   centroids=centroids,  # The centroids of collisions, chosen through kmeans
                                   labels=labels,  # The list of which collisions belong to which centroid
                                   only_show_num_collisions=0,
                                   # Randomly pick only some collisions to show, for a cleaner map
                                   output_file_name="july_2017"))

    # June 2018
    # Calculate severity score like low cost for injury and high cost for death penalty
    collisions_in_brooklyn_june_2018 = visualizer_object.severity_score_measure(df_june_2018)
    # Run kmeans with 15 centroids, find 15 'collision hotspots'
    centroids, labels = visualizer_object.get_centroids_clusters(collisions_in_brooklyn_june_2018, 15)

    print(visualizer_object.map_collisions_visualize(collisions_in_brooklyn_june_2018,  # The list of collisions to plot
                                   centroids=centroids,  # The centroids of collisions, chosen through kmeans
                                   labels=labels,  # The list of which collisions belong to which centroid
                                   only_show_num_collisions=0,
                                   # Randomly pick only some collisions to show, for a cleaner map
                                   output_file_name="june_2018"))

    # July 2018
    # Calculate severity score like low cost for injury and high cost for death penalty
    collisions_in_brooklyn_july_2018 = visualizer_object.severity_score_measure(df_july_2018)
    # Run kmeans with 15 centroids, find 15 'collision hotspots'
    centroids, labels = visualizer_object.get_centroids_clusters(collisions_in_brooklyn_july_2018, 15)

    print(visualizer_object.map_collisions_visualize(collisions_in_brooklyn_july_2018,  # The list of collisions to plot
                                   centroids=centroids,  # The centroids of collisions, chosen through kmeans
                                   labels=labels,  # The list of which collisions belong to which centroid
                                   only_show_num_collisions=0,
                                   # Randomly pick only some collisions to show, for a cleaner map
                                   output_file_name="july_2018"))


    #Visualize number of pedestrains killed/injured - for pedestrains safety
    # visualizer_object.view_number_of_pedestrains_killed(df_2017,df_2018)


    # Visualize different factors - pedestrians injured and killed for
    # june july 2017, 2018
    pedestrains_injured = visualizer_object.get_count_pedestrains(df_june_2017, 'NUMBER OF PEDESTRIANS INJURED', 'June2017')

    pedestrains_killed = visualizer_object.get_count_pedestrains(df_june_2017, 'NUMBER OF PEDESTRIANS KILLED', 'June2017')

    pedestrains_injured = visualizer_object.get_count_pedestrains(df_july_2017, 'NUMBER OF PEDESTRIANS INJURED',
                                                'July2017')
    pedestrains_killed = visualizer_object.get_count_pedestrains(df_july_2017, 'NUMBER OF PEDESTRIANS KILLED', 'July2017')

    pedestrains_injured = visualizer_object.get_count_pedestrains(df_june_2018, 'NUMBER OF PEDESTRIANS INJURED',
                                                'June2018')
    pedestrains_killed = visualizer_object.get_count_pedestrains(df_june_2018, 'NUMBER OF PEDESTRIANS KILLED', 'June2018')

    pedestrains_injured = visualizer_object.get_count_pedestrains(df_july_2018, 'NUMBER OF PEDESTRIANS INJURED',
                                                'July2018')
    pedestrains_killed = visualizer_object.get_count_pedestrains(df_july_2018, 'NUMBER OF PEDESTRIANS KILLED', 'July2018')


"""
Output:
/home/pranavmrane/PycharmProjects/BDA_Project2/venv/bin/python /home/pranavmrane/PycharmProjects/BDA_Project2/Code/ControlMenu.py
(39642, 30)
(39323, 30)
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
2017 Jul 01 : 582
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
('smallest non-zero count', 8.577404340105357e-05)
('max count:', 2.516574822254173)
('smallest non-zero count', 8.577404340105357e-05)
('max count:', 2.516574822254173)
('smallest non-zero count', 1.791063608477192e-08)
('max count:', 6.047982810383686)
('smallest non-zero count', 8.577404340105357e-05)
('max count:', 6.843781910490505)
[[ 40.62346919 -73.98139866]
 [ 40.6784721  -73.93415766]]
[[ 40.61731894 -73.98160033]
 [ 40.66089519 -73.90802727]
 [ 40.69071793 -73.9651277 ]]
[[ 40.63013842 -74.00399175]
 [ 40.69184638 -73.96374477]
 [ 40.61126084 -73.95343023]
 [ 40.66593477 -73.90553682]]
[[ 40.67809732 -73.93274263]
 [ 40.62468286 -73.98262821]]
[[ 40.61771301 -73.98337641]
 [ 40.66004173 -73.90672634]
 [ 40.69048222 -73.96389588]]
[[ 40.64053635 -74.00631403]
 [ 40.66364695 -73.90419809]
 [ 40.69288904 -73.9595966 ]
 [ 40.61001372 -73.95982151]]
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
