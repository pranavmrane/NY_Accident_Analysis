"""
CSCI720 Project 2
Name: DataCluster.py
Program to evaluate , provide evidence of learning and
understanding of assignment on NYC Traffic Collisions
Date: 1st Dec 2018
Authors:Priyanka Patil (pxp1439@rit.edu)
Authors: Pranav Rane (pmr5279@rit.edu)
"""

#imports
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geoplotlib as geoplotlib
from geoplotlib.utils import BoundingBox


class DataCluster:
    # Class Should contain all the code to make and visualize Clusters
    def __init__(self):
        pass

    def factor_vehicle_accidents_graph(self, df_june_2017, df_july_2017,
                                       df_june_2018, df_july_2018):
        """
        Plot of vehicle accidents count against each borough Brooklyn
        in various years to visualise accident distributions / changes
        :param df_june_2017: Accidents Occurences in June 2017
        :param df_july_2017: Accidents Occurences in July 2017
        :param df_june_2018: Accidents Occurences in June 2018
        :param df_july_2018: Accidents Occurences in July 2018
        :return: None
        """
        plt.title("Vehicle Accidents Occurance Distribution ")
        df = pd.DataFrame(
            {'x': range(1, 54),
             'Brooklyn June 2017': df_june_2017,
             'Brooklyn July 2017': df_july_2017,
             'Brooklyn June 2018': df_june_2018,
             'Brooklyn July 2018': df_july_2018})

        plt.plot('x', 'Brooklyn June 2017', data=df, color='blue')
        plt.plot('x', 'Brooklyn July 2017', data=df, color='red')
        plt.plot('x', 'Brooklyn June 2018', data=df, color='green')
        plt.plot('x', 'Brooklyn July 2018', data=df, color='yellow')
        plt.xticks(range(1, 54), df.index, rotation='vertical')
        plt.legend()
        plt.show()

    def perform_clustering_kMeans(self, df, title):
        """
        k-Means Clsutering on Brookyln to get clusters of
        various sizes and splitting into 4 clusters
        :param df: dataset having Latitude, Longitude, Severity Score
        :param title: Title of plot to show
        :return: None
        """
        # # filtering the dataset and save the coordinates
        data = df.filter(items=['LATITUDE', 'LONGITUDE', 'SEVERITY SCORE'])
        # turn into a numpy array
        kmeans_array = np.array(data)

        # calculating the coordinates of the centroids of the clusters for 2,3 and 4 clusters
        for i in range(2, 5):
            kmeans2 = KMeans(n_clusters=i, random_state=0).fit(data)
            print(kmeans2.cluster_centers_)

        # Lets choose k from 2 ,3,4 clusters to visualize results for data distribution
        for i in range(2, 5):
            results = KMeans(n_clusters=i, random_state=0).fit_predict(
                kmeans_array)
            kmeans2 = KMeans(n_clusters=i, random_state=0).fit(data)
            plt.scatter(kmeans_array[:, 1], kmeans_array[:, 0], c=results)
            plt.scatter(kmeans2.cluster_centers_[:, 1],
                        kmeans2.cluster_centers_[:, 0], c='red')
            plt.title(title)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()

    def visualize_heatmap_for_data(self, df):
        """
        Normally visualize accidents distributions across borough
        :param df: Dataset
        :return: None
        """
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

    def remove_unspecified(self, dataset):
        """
        Remove values in dataset having unspecified values
        since it will result in different values if not removed.
        This is due to mystery result, so to avoid this, it is done removing
        unspecified values.
        :param dataset: Dataset having all factors
        :return: count of factor 1 values
        """
        vehicle_factor_count = dataset.loc[
            (dataset['CONTRIBUTING FACTOR VEHICLE 1'] != 'Unspecified')]
        vehicle_factor_count = vehicle_factor_count[
            'CONTRIBUTING FACTOR VEHICLE 1'].value_counts()

        return vehicle_factor_count

