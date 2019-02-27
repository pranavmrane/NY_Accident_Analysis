"""
CSCI720 Project 2
Name: DataVisualizer.py
Program to evaluate , provide evidence of learning and
understanding of assignment on NYC Traffic Collisions
Date: 1st Dec 2018
Authors:Priyanka Patil (pxp1439@rit.edu)
Authors: Pranav Rane (pmr5279@rit.edu)
"""
import collections
import operator
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import kmeans2
import warnings
import datetime
import numpy as np
import os
from os import path
from colour import Color
import gmplot as gmplot

#MapBox API Token
mapbox_access_token = 'pk.eyJ1IjoiYnVtYmxlYmVlY29kZXIiLCJhIjoiY2pwaGU2cjR2MHZxbjNxb2JvZzI5aThoNiJ9.B1uQr6MHRi3s2llorjLc9A'

class DataVisualizer:
    # Class Should contain all the code to make graphs
    # Borough selection decision should also be made here
    def __init__(self):
        plt.style.use('ggplot')
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def make_date_time(self, dataset):
        """
        Change time format to pandas Type
        :param dataset: Dataset having date and time attributes
        :return: Reformatted dataset
        """
        dataset['DATETIME'] = pd.to_datetime(
            dataset['DATE'] + ' ' + dataset['TIME'])

        return dataset

    def plot_hourly_distribution_graph(self, time, title):
        """
        Shows hours - distribution across various accidents happening
        Use Binning of size 40.
        :param time: Time values count (in hours)
        :param title: Title of plot to show
        :return: None
        """
        plt.hist(time, bins=47)
        plt.xlabel("Hours of the accident")
        plt.ylabel("Accident Count")
        plt.title(title)
        plt.show()

    def plot_weekly_distribution_graph(self, week, title):
        """
        Shows a weekly distribution ( weekdays and weekends) of
        accidents happening acroos Brooklyn
        :param week: week values counts
        :param title: Title to plot to show
        :return: None
        """
        plt.title(title)
        plt.xlabel("Days In The Week")
        plt.ylabel("Accident Count")
        plt.bar(range(len(week.keys())), week.values())
        weeks = ['Mon', 'Tues', 'Wedn', 'Thurs', 'Fri', 'Sat', 'Sun']
        plt.xticks(range(len(week.keys())), weeks)
        plt.show()

    def less_and_high_accident_occurrences_with_dates(self,
                                                      no_of_accidents_dates,
                                                      borough_accidents, year):
        """
        Display in consoe the number of less and high accidents count
        with their corresponding dates and year
        :param no_of_accidents_dates: List of accidents with dates counts
        :param borough_accidents: List of accidents counts in Brooklyn
        :param year: Years - 2017, 2018
        :return: None
        """
        borough_accident_ = collections.Counter(borough_accidents)
        sorted_by_num_of_accidents = \
            sorted(no_of_accidents_dates.items(), key=operator.itemgetter(1))

        print("Less Accidents in ", year, ":")
        less_Accidents = sorted_by_num_of_accidents[0:10]

        for date_ in less_Accidents:
            print(date_[0].strftime("%Y %b %d"), ":", date_[1])

        print("High Accidents in", year, ":")
        high_Accidents = sorted_by_num_of_accidents[-10:]

        for date_ in high_Accidents:
            print(date_[0].strftime("%Y %b %d"), ":", date_[1])

        # View Accident count distributions across various boroughs of NYC
        plt.bar(range(5),
                borough_accident_.values())
        plt.xticks(range(5)
                   , borough_accident_.keys())
        plt.xlabel("Borough Names")
        plt.ylabel("Accidents Count")

        # plt.gcf().set_size_inches(11,11)
        title = "Accident Counts for June-July" + year
        plt.title(title)
        plt.show()


    def get_centroids_clusters(self, df, centres):
        """
        Get centroids of clusters usinng k-MEans
        :param df: dataset with Latitude and Longitude
        :param centres: k value
        :return: centroids and labels of clusters formed
        """
        location_data = df.loc[:, ['LATITUDE', 'LONGITUDE']]
        # Perform kMeans on data
        centroids, labels = kmeans2(location_data, centres)
        return centroids, labels

    def severity_score_measure(self, df):
        """
        Assign severity score to create color range
        Red- High accident severity
        Yellow- Low accident severity
        :param df: Dataset
        :return: Dataset with assigned severity score for every factors
        """
        INJURY = 10
        DEATH = 100
        df['SEVERITY SCORE'] = 1
        # For each injury, add to the score of the collision
        df['SEVERITY SCORE'] += (df.loc[:, 'NUMBER OF PERSONS INJURED'] * INJURY)
        # For each death, add to the score of the collision
        df['SEVERITY SCORE'] += (df.loc[:, 'NUMBER OF PERSONS KILLED'] * DEATH)
        return df

    def map_collisions_visualize(self, data,
                                 features=[],
                                 heatmap=True,
                                 centroids=None,
                                 labels=None,
                                 color_collisions_based_on_labels=None,
                                 only_show_num_collisions=None,
                                 output_file_name='map',
                                 lat=40.77,
                                 lon=-74.009725,
                                 zoom=10):
        """
        Create collisions (high to low intensity) on heatmap. Plot their
        collision hotspots and centroids.
        :param data: Dataset
        :param features: features of dataset if we choose
        :param heatmap: Boolean to display intensity colors or not
        :param centroids: centroids of collisions clusters
        :param labels: labels of collisions clusters
        :param color_collisions_based_on_labels: Colors for high to low intensity
        :param only_show_num_collisions: Value to show number of collision data points
        :param output_file_name: file name for creating .html and .js files
        :param lat: latitude for plotting
        :param lon: longitude for plotting
        :param zoom: zoom feature while opening map .html file
        :return: map url for viewing collisions hotspots of varying intensites
        """

        prime_colors = ['Red', 'Blue', 'Purple', 'Green', 'Yellow', 'Cyan', 'Magenta', 'Black', 'White']

        # Creating map files in form of .html
        self.create_map_html_file(output_file_name)
        self.create_map_script_file(output_file_name, lat, lon, zoom)

        # create duplicate of data frame
        collisions = data.copy().reset_index(drop=True)

        if labels is not None:
            # Append the labels of collision data
            collisions = collisions.assign(LABELS=labels)

        # Filter the collisions based on the list of features that were given
        # If any features present
        for feature in features:
            collisions = collisions.loc[collisions[feature] == 1, :].reset_index(drop=True)

        # After filtering, throw away the unneeded features
        if labels is not None:
            collisions = collisions.loc[:, ['LATITUDE', 'LONGITUDE', 'SEVERITY SCORE', 'LABELS']]
        else:
            collisions = collisions.loc[:, ['LATITUDE', 'LONGITUDE', 'SEVERITY SCORE']]

        # Pick any number of collisions if specified in randomly
        num_collisions_total = collisions.shape[0]
        if only_show_num_collisions and only_show_num_collisions < num_collisions_total:
            collisions_random = collisions.loc[np.random.choice(num_collisions_total,
                                                                only_show_num_collisions,
                                                                replace=False), :]
            collisions = collisions_random[:only_show_num_collisions].reset_index(drop=True)

        if heatmap:  # Plot with heatmap
            # Get the color range
            unique_severity_scores = sorted(collisions.loc[:, 'SEVERITY SCORE'].unique())
            num_unique_severity_scores = len(unique_severity_scores)
            colors = list(Color('Yellow').range_to('Red', num_unique_severity_scores))

            # Plot each severity level with its own color
            for index, score in enumerate(unique_severity_scores):
                    collisions_with_score = collisions.loc[collisions['SEVERITY SCORE'] == score, :]
                    # print(collisions_with_score)

                    self.writing_points_file(output_file_name, collisions_with_score, colors[index], 50)
        elif labels is not None:  # Plot with color code
            # Plot each center's points
            color_to_use = Color('Blue')
            for center in range(len(centroids)):
                if color_collisions_based_on_labels and len(centroids) <= len(prime_colors):
                    color_to_use = Color(prime_colors[center])
                collisions_with_label = collisions.loc[collisions.loc[:, 'LABELS'] == center, :]
                self.writing_points_file(output_file_name, collisions_with_label, color_to_use, 50)
        else:
            # Plot simple graph
            self.writing_points_file(output_file_name, collisions, Color('Blue'), 50)

        # if centroids is not None:
        #     # Plot the centroids
        #     self.writing_points_file(output_file_name, centroids, Color('Orange'), 150)

        # Generate map URL
        map_url = self.get_map_output_url(output_file_name)

        return map_url

    def get_map_output_url(self,file_name):
        """
        Get map output url by inputting file name and path
        :param file_name: file name that is called
        :return: file name of generated html (map)
        """
        return 'file://%s' % path.join(os.getcwd(), file_name + '.html')

    def create_map_html_file(self, file_name):
        """
        Write a html script for plotting data points and varying intensities
        of collisions to view
        :param file_name: generated file name
        :return:
        """
        page = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>TITLE</title>
                    <meta charset="utf-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">

                    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.0.2/dist/leaflet.css" />
                    <script src="https://unpkg.com/leaflet@1.0.2/dist/leaflet.js"></script>
                </head>
                <body>
                    <div id="mapid" style="min-width: 800px; min-height: 800px;"></div>
                    <script src="SCRIPT"></script>
                </body>
            </html>
        """

        page = page.replace("TITLE", file_name)
        page = page.replace("SCRIPT", file_name + '.js')

        with open(file_name + '.html', "w") as map_file:
            map_file.write(page)

    def create_map_script_file(self, file_name, lat, lon, zoom):
        """
        Writing .js script and access token granted from MAPBOX API for plotting
        onto map that can be called by .html file
        :param file_name: generated file name
        :param lat: latitude
        :param lon: longitude
        :param zoom: zoom feature
        :return:
        """
        script = """    
        var mymap = L.map('mapid').setView([LAT, LON], ZOOM);

        L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token=ACC_TOKEN', {
            maxZoom: 18,
            attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, ' +
                '<a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
                'Imagery © <a href="http://mapbox.com">Mapbox</a>',
            id: 'mapbox.streets'
        }).addTo(mymap);
        """
        script = script.replace("LAT", str(lat))
        script = script.replace("LON", str(lon))
        script = script.replace("ZOOM", str(zoom))
        script = script.replace("ACC_TOKEN", mapbox_access_token)

        with open(file_name + '.js', "w") as script_file:
            script_file.write(script)

    def writing_points_file(self, file_name, points, color, size):
        """
        Appending datapoints to .js script for plotting onto the map that
         can be called by .html file
        :param file_name:
        :param points:
        :param color:
        :param size:
        :return:
        """
        point_js = """
        L.circle([LAT, LON], SIZE, {
            color: 'COLOR',
            fillColor: 'COLOR',
            fillOpacity: 0.5
        }).addTo(mymap);
        """

        point_js = point_js.replace('SIZE', str(size))
        point_js = point_js.replace('COLOR', str(color.hex))

        if type(points) == pd.DataFrame:
            points = points.loc[:, ['LATITUDE', 'LONGITUDE']].values

        with open(file_name + '.js', "a") as script_file:
            for lat, lon in points:
                new_point_js = point_js.replace('LAT', str(lat))
                new_point_js = new_point_js.replace('LON', str(lon))
                script_file.write(new_point_js)


    def view_number_of_pedestrains_killed(self, df_2017, df_2018):

        # df_2017_cleaned = df_2017['NUMBER OF PEDESTRIANS KILLED'].replace(np.NaN, 'OTHER')
        kill_count = {}
        for val in df_2017['DATE'].unique():
            temp = df_2017.loc[df_2017['DATE'].isin([val])]
            kill_count[val] = int(temp['NUMBER OF PEDESTRIANS KILLED'].sum(axis=0))

        plt.bar(kill_count.keys(), kill_count.values())
        # print(kill_count.keys(), kill_count.values())
        plt.xlabel("Dates")
        plt.ylabel("Pedestrains Count")
        plt.title("Number of Pedestrains Killed under June & July 2018")
        plt.show()

        # persons_killed = df_2018['NUMBER OF PEDESTRIANS KILLED'].replace(np.NaN, 'OTHER')
        kill_count = {}
        for val in df_2018['DATE'].unique():
            temp = df_2018.loc[df_2018['DATE'].isin([val])]
            kill_count[val] = int(temp['NUMBER OF PEDESTRIANS KILLED'].sum(axis=0))

        plt.bar(kill_count.keys(), kill_count.values())
        # print(kill_count.keys(), kill_count.values())
        plt.xlabel("Dates")
        plt.ylabel("Pedestrains Count")
        plt.title("Number of Pedestrains Killed under June & July 2017")
        plt.show()

    def get_count_pedestrains(self, df, topic, monthyear):
        """
        View places that got pedestrains killed and injured for
        various years and show changes in map (.html)
        :param df: dataset
        :param topic: title of map
        :param monthyear: month and year in which case is happening
        :return: None
        """
        list_count = []
        for index, row in df.iterrows():
            each_value = row[topic]
            lat = row['LATITUDE']
            lng = row['LONGITUDE']
            if each_value > 0:
                lat = float(lat)
                lng = float(lng)
                list_count.append(((lng, lat), each_value))
        print(len(list_count), ' collisions has ', topic.lower(), 'in', monthyear)
        count = collections.Counter(list_count)

        min_lat, max_lat, min_lon, max_lon = \
            min(df['LATITUDE']), max(df['LATITUDE']), \
            min(df['LONGITUDE']), max(df['LONGITUDE'])

        # Create empty map with zoom level 16
        mymap = gmplot.GoogleMapPlotter(
            min_lat + (max_lat - min_lat) / 2,
            min_lon + (max_lon - min_lon) / 2,
            16)

        newdf = pd.DataFrame(columns=['lat', 'lon'])
        for k, v in count.most_common(5):
            latlng, value = k[0], k[1]
            newdf = newdf.append({
                'lon': latlng[0],
                'lat': latlng[1]

            }, ignore_index=True)


        mymap.scatter(newdf['lat'], newdf['lon'], 'blue', edge_width=4)
        mymap.draw(topic.lower() + ' '+ monthyear + '.html')


    def high_processing(self, df_2017, df_2018):
        """
        Parsing, Retrieval of 2017 and 2018 dataset
        Plot hourly Distribution and weekly Distribution acrros various years
        Show counts of high and low accident counts with their corresponding dates
        :param df_2017:Dataset under June-July 2017
        :param df_2018:Dataset unnder June-July 2018
        :return:
        """
        # Parsing and retrieving from 2017 Dataset
        all_accident_borough_2017 = []
        borough_accident_2017 = []
        no_of_accidents_dates = {}
        time_hour_accident = []
        accidents_weekdays = {}
        location = []
        for index, row in df_2017.iterrows():
            lat = float(row['LATITUDE'])
            long = float(row['LONGITUDE'])
            date = row['DATE']
            extracted_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            weekly_date = datetime.datetime.strptime(date,
                                                     "%Y-%m-%d").weekday()
            time_hours = row['TIME']
            try:
                no_of_accidents_dates[extracted_date] += 1
                accidents_weekdays[weekly_date] += 1
            except KeyError:
                no_of_accidents_dates[extracted_date] = 1
                accidents_weekdays[weekly_date] = 1

            all_accident_borough_2017.append(row)
            borough_accident_2017.append(row['BOROUGH'])
            location.append((lat, long))
            # Time
            if ':' in time_hours:
                time_hour = time_hours.split(":")
                time_hour_accident.append(int(time_hour[0]))

        locations_counter = collections.Counter(location)
        # plotting hourly distribution of accidents  (2017)
        self.plot_hourly_distribution_graph(time_hour_accident,
                                            "Hourly Distribution of Accidents in June-July 2017")

        # To sort and then count number of accidents weekly
        accidents_weekdays = sorted(accidents_weekdays.items())
        weekly_accidents_count = collections.OrderedDict(accidents_weekdays)
        # Plotting weekly accidents count
        self.plot_weekly_distribution_graph(weekly_accidents_count,
                                            "Weekly Distribution of June-July 2017")

        # Lets try to see dates having high and low vehicles collisions (2017)
        # To count number of high accidents and low accidents
        self.less_and_high_accident_occurrences_with_dates(
            no_of_accidents_dates,
            borough_accident_2017,
            "2017")

        # Parsing and retrieving from 2018 Dataset
        all_accident_borough_2018 = []
        borough_accident_2018 = []
        for index, row in df_2018.iterrows():
            all_accident_borough_2018.append(row)
            borough_accident_2018.append(row['BOROUGH'])
            time_hours = row['TIME']
            # Time
            if ':' in time_hours:
                time_hour = time_hours.split(":")
                time_hour_accident.append(int(time_hour[0]))

        # plotting hourly distribution of accidents  (2018)
        self.plot_hourly_distribution_graph(time_hour_accident,
                                            "Hourly Distribution of Accidents of June-July 2018")

        # Plotting weekly distribution of accidents (2018):
        self.plot_weekly_distribution_graph(weekly_accidents_count,
                                            "Weekly Distribution of June-July 2018")

        # View Less and High Accident Occurances with dates in 2018:
        self.less_and_high_accident_occurrences_with_dates(
            no_of_accidents_dates,
            borough_accident_2018,
            "2018")


        # We decided to choose borough - 'Brooklyn'
        df_2017 = df_2017[df_2017['BOROUGH'] == 'BROOKLYN']
        df_2018 = df_2018[df_2018['BOROUGH'] == 'BROOKLYN']

        # Heatmap Showing Brooklyn for June July 2017 vs Brooklyn June July 2018:
        heatmap, x_edges, y_edges = np.histogram2d(df_2017['LONGITUDE']
                                                   , df_2017['LATITUDE'],
                                                   bins=100)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='Greys')
        plt.colorbar()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis(aspect='auto')
        plt.grid(False)
        plt.title('Heat map of June & July 2017')
        plt.show()

        heatmap, xedges, yedges = np.histogram2d(df_2018['LONGITUDE']
                                                 , df_2018['LATITUDE'],
                                                 bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='Greys')
        plt.colorbar()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis(aspect='auto')
        plt.grid(False)
        plt.title('Heat map of June & July 2018')
        plt.show()


        return df_2017, df_2018
