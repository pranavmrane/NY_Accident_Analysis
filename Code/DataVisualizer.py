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

#MapBox API Token
mapbox_access_token = 'pk.eyJ1IjoiYnVtYmxlYmVlY29kZXIiLCJhIjoiY2pwaGU2cjR2MHZxbjNxb2JvZzI5aThoNiJ9.B1uQr6MHRi3s2llorjLc9A'

class DataVisualizer:
    # Class Should contain all the code to make graphs
    # Borough selection decision should also be made here
    def __init__(self):
        plt.style.use('ggplot')
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # Priya

    def make_date_time(self, dataset):
        dataset['DATETIME'] = pd.to_datetime(
            dataset['DATE'] + ' ' + dataset['TIME'])

        return dataset

    def plot_hourly_distribution_graph(self, time, title):
        plt.hist(time, bins=47)
        plt.xlabel("Hours of the accident")
        plt.ylabel("Accident Count")
        plt.title(title)
        plt.show()

    def plot_weekly_distribution_graph(self, week, title):
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
        # TODO: Fix plt title
        # plt.gcf().set_size_inches(11,11)
        # plt.title("Accident Counts for June-July", year)
        plt.show()


    def get_centroids_clusters(self, df, centres):
        location_data = df.loc[:, ['LATITUDE', 'LONGITUDE']]
        # Perform kMeans on data
        centroids, labels = kmeans2(location_data, centres)
        return centroids, labels

    def severity_score_measure(self, df):
        INJURY = 1
        DEATH = 10
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
            print(feature)
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

        if centroids is not None:
            # Plot the centroids
            self.writing_points_file(output_file_name, centroids, Color('Orange'), 150)

        # Generate map URL
        map_url = self.get_map_output_url(output_file_name)

        return map_url

    def get_map_output_url(self,file_name):
        return 'file://%s' % path.join(os.getcwd(), file_name + '.html')

    def create_map_html_file(self, file_name):
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
                    <div id="mapid" style="min-width: 800px; min-height: 600px;"></div>
                    <script src="SCRIPT"></script>
                </body>
            </html>
        """

        page = page.replace("TITLE", file_name)
        page = page.replace("SCRIPT", file_name + '.js')

        with open(file_name + '.html', "w") as map_file:
            map_file.write(page)

    def create_map_script_file(self, file_name, lat, lon, zoom):

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
        persons_killed = df_2017['NUMBER OF PERSONS KILLED'].replace(np.NaN, 'OTHER')
        kill_count = {}
        for val in df_2017['DATE'].unique():
            temp = df_2017.loc[df_2017['DATE'].isin([val])]
            kill_count[val] = int(temp['NUMBER OF PERSONS KILLED'].sum(axis=0))

        plt.bar(kill_count.keys(), kill_count.values())
        print(kill_count.keys(), kill_count.values())
        plt.show()

        persons_killed = df_2018['NUMBER OF PERSONS KILLED'].replace(np.NaN, 'OTHER')
        kill_count = {}
        for val in df_2018['DATE'].unique():
            temp = df_2018.loc[df_2018['DATE'].isin([val])]
            kill_count[val] = int(temp['NUMBER OF PERSONS KILLED'].sum(axis=0))

        plt.bar(kill_count.keys(), kill_count.values())
        print(kill_count.keys(), kill_count.values())
        plt.show()

    def high_processing(self, df_2017, df_2018):
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
                                            "Hourly Distribution of Accidents in June-July 2018")

        # Plotting weekly distribution of accidents (2018):
        self.plot_weekly_distribution_graph(weekly_accidents_count,
                                            "Weekly Distribution of June-July 2018")

        # View Less and High Accident Occurances with dates in 2018:
        self.less_and_high_accident_occurrences_with_dates(
            no_of_accidents_dates,
            borough_accident_2018,
            "2018")

        # Heatmap
        heatmap, x_edges, y_edges = np.histogram2d(df_2017['LONGITUDE']
                                                   , df_2017['LATITUDE'],
                                                   bins=100)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='copper')
        plt.colorbar()
        plt.axis(aspect='image')
        plt.title('Heat map of June & july 2017')
        plt.show()

        heatmap, xedges, yedges = np.histogram2d(df_2018['LONGITUDE']
                                                 , df_2018['LATITUDE'],
                                                 bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='copper')
        plt.colorbar()
        plt.axis(aspect='image')
        plt.title('Heat map of June & july 2018')
        plt.show()

        # visualize different factors causing accidents groupby boroughs
        df_2017['COUNTER'] = 1
        df_2018['COUNTER'] = 1
        group_2017 = df_2017.groupby(['BOROUGH', 'NUMBER OF PERSONS KILLED'])[
            'COUNTER'].sum()
        group_2018 = df_2018.groupby(['BOROUGH', 'NUMBER OF PERSONS KILLED'])[
            'COUNTER'].sum()
        # print(group_2017.to_string())
        # plt.title("View borouoghs 2018")
        # group_2017.plot.bar()
        # plt.show()
        # print(group_2018.to_string())
        # plt.title("View borouoghs 2018")
        # group_2018.plot.bar()
        # plt.show()

        # We decided to choose borough - 'Brooklyn'
        df_2017 = df_2017[df_2017['BOROUGH'] == 'BROOKLYN']
        df_2018 = df_2018[df_2018['BOROUGH'] == 'BROOKLYN']

        return df_2017, df_2018
