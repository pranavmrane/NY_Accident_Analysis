import collections
import operator
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import kmeans2
import warnings
import datetime
import numpy as np


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

    # Optional
    def get_centroids_clusters(self, df, centres):
        location_data = df.loc[:, ['LATITUDE', 'LONGITUDE']]
        # Perform kMeans on data
        centroids, labels = kmeans2(location_data, centres)
        return centroids, labels

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
