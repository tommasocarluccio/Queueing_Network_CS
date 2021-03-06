import json
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from sklearn import neighbors

class Scenario:
    def __init__(self, scenario_conf):
        self.json_file=scenario_conf
        self.scenario=json.load(open(self.json_file,"r"))
        self.battery_capacity_reduced=self.scenario["battery_capacity"]*(self.scenario["max_charging_th"]-self.scenario["min_charging_th"])
        self.dist_autonomy=self.battery_capacity_reduced/self.scenario["av_consumption"] #km
        self.trips_autonomy=self.dist_autonomy/self.scenario["av_trip_length"]
        self.outlet_rate=self.scenario["outlet_power"]/self.battery_capacity_reduced #hourly rate of charging operation
        self.n_charging_stations=self.scenario["n_charging_stations"]
        self.delay_queue=self.scenario['delay_queue']
        self.av_trip_time=self.scenario['av_trip_time']
        self.charging_policy=self.scenario['charging_policy']
        self.reloc_after_charging=self.scenario['reloc_after_charging']
        self.n_top_zones=self.scenario['n_top_zones']
        self.n_vehicles=self.scenario["n_vehicles"]
        #data
        self.bookings_df=pd.read_csv("trips_with_zone_id.csv")
        try:
            self.city_grid = pd.read_pickle("new_grid.pickle")
        except:
            self.city_grid=None
        self.hour_of_day=np.arange(self.scenario['start_hour'],self.scenario['end_hour'])
        #from OD
        self.n_zones, self.zones_id_dict, self.OD_matrix, self.zones_service_rates, self.trips_data=self.get_hour_OD()
        if self.n_charging_stations!=0:
            self.zones_with_charging_stations=self.place_CS_by()
        self.n_servers, self.service_rates=self.build_service_rate()

    def get_hour_OD(self):
        print("\nNumber of entries in original dataframe: ", len(self.bookings_df.index))
        #filter bookings data
        self.bookings_df=self.bookings_df.loc[(self.bookings_df['duration']>3*60) & (self.bookings_df['duration']<90*60)]
        self.bookings_df=self.bookings_df.loc[self.bookings_df['distance']>500]
        print("Number of entries in dataframe (real trips): ", len(self.bookings_df.index))
        self.bookings_df['start_date'] = pd.to_datetime(self.bookings_df['init_time'],unit='s').dt.date
        self.bookings_df['start_hour'] = pd.to_datetime(self.bookings_df['init_time'],unit='s').dt.hour
        if self.scenario['week_day']:
            self.bookings_df=self.bookings_df.loc[self.bookings_df['weekday'].isin([0,1,2,3,4])]
        try:
            self.bookings_df=self.bookings_df.loc[self.bookings_df['start_hour']==self.hour_of_day]
            interval_size=1
            print(f"Single hour bookings h:{self.hour_of_day}")
        except:
            self.bookings_df=self.bookings_df.loc[self.bookings_df['start_hour'].isin(self.hour_of_day)]
            interval_size=len(self.hour_of_day)
            print(f"Interval of hours bookings h:{self.hour_of_day}")
        print("Number of entries in filtered dataframe: ", len(self.bookings_df.index))
        #get zones
        zones_id=np.unique(self.bookings_df[['origin_id', 'destination_id']].values).astype(int)
        zones_id=np.sort(zones_id)
        n_zones=zones_id.size
        #get grid with only valid zones
        try:
            self.city_grid=self.city_grid.loc[self.city_grid['zone_id'].isin(zones_id)]
        except:
            pass
        #create dicr of zones for OD-grid
        #map index of OD matrix to zone indexes
        zones_id_dict={}
        for i in range(n_zones):
            zones_id_dict[i]=zones_id[i]
        inv_zones_id_dict = {v: k for k, v in zones_id_dict.items()}
        
        #df for OD
        bookings_data_OD=self.bookings_df[['origin_id','destination_id']]
        bookings_data_OD_grouped=bookings_data_OD.groupby(['origin_id','destination_id']).size().reset_index(name='counts')
        #bookings_df_rate=bookings_df_rate.reindex(zones_id, fill_value=0)
        OD_matrix=np.zeros((n_zones,n_zones))
        tot_departure_per_zone=np.zeros(n_zones)
        tot_arrival_per_zone=np.zeros(n_zones)
        for index, row in bookings_data_OD_grouped.iterrows():
            OD_matrix[inv_zones_id_dict[row['origin_id']],inv_zones_id_dict[row['destination_id']]]=row['counts']
        for i in range (n_zones):
            if np.sum(OD_matrix[i,:])==0:
                OD_matrix[i][i]=1
            tot_departure_per_zone[i]=np.sum(OD_matrix[i,:])
            tot_arrival_per_zone[i]=np.sum(OD_matrix[:,i])

        trips_data=pd.DataFrame(list(tot_departure_per_zone),index=self.city_grid.zone_id, columns=['departure_per_zone'])
        trips_data['arrival_per_zone']=list(tot_arrival_per_zone)

        OD_matrix=OD_matrix/OD_matrix.sum(axis=1)[:,None]

        #df for service rates
        bookings_df_rate=self.bookings_df.groupby(['origin_id','start_date','start_hour']).size().reset_index(name='counts')
        #print(bookings_df_rate)
        bookings_df_rate=bookings_df_rate.groupby(['origin_id','start_date']).agg(date_mean=("counts",'sum'))
        bookings_df_rate['date_mean']=bookings_df_rate['date_mean']/interval_size
        #print(bookings_df_rate)
        bookings_df_rate=bookings_df_rate.groupby('origin_id').agg(mean_service_time=('date_mean','mean'))
        #print(bookings_df_rate)
        bookings_df_rate=bookings_df_rate.reindex(zones_id, fill_value=0.1)
        #self.zones_service_rates=np.zeros(n_zones)
        zones_service_rates=np.array(bookings_df_rate['mean_service_time'])
        
        return n_zones, zones_id_dict, OD_matrix, zones_service_rates, trips_data

    def place_CS_by(self):
        order_by=self.scenario["order_CS_position_by"]
        if order_by=='random':
            zones_with_charging_stations=np.sort(np.random.choice(self.n_zones,self.n_charging_stations,replace=False))    
        else:
            complete=False
            zones_with_charging_stations=[]
            #neighbors_delta=[1,-1,40,-40,39,41,-39,-41]
            neighbors_delta=[1,-1,40,-40,39,41,-39,-41,2,-2,41,38,-41,-38,-82,-81,-80,-79,-78,78,79,80,81,82,42,-42]
            #neighbors_delta=[]
            while not complete:
                ordered_df_column=self.trips_data.sort_values(by=order_by, ascending=False)[[order_by]]
                inv_zones_id_dict = {v: k for k, v in self.zones_id_dict.items()}
                grid_with_charging_stations=[]
                flag=0
                for idx, row in ordered_df_column.iterrows():
                    for i in grid_with_charging_stations:
                        if idx in [i+j for j in neighbors_delta]:
                            flag=1
                            break
                    if flag==0:
                        grid_with_charging_stations.append(idx)
                        if len(grid_with_charging_stations)==self.n_charging_stations:
                            break
                    flag=0
                #print(grid_with_charging_stations)
                zones_with_charging_stations=[inv_zones_id_dict[i] for i in grid_with_charging_stations]
                print(len(zones_with_charging_stations))
                if len(zones_with_charging_stations)==self.n_charging_stations:
                    complete=True
                else:
                    neighbors_delta=[1,-1,40,-40,39,41,-39,-41]
        return zones_with_charging_stations

    def build_service_rate(self):
        service_rates=np.ones(self.n_zones+self.n_charging_stations)
        service_rates[0:self.n_zones]=self.zones_service_rates
        n_servers=np.ones(self.n_zones+self.n_charging_stations)
        if self.n_charging_stations!=0:
            service_rates[self.n_zones:self.n_zones+self.n_charging_stations]=(np.ones(self.n_charging_stations)*self.outlet_rate)
            n_servers[self.n_zones:self.n_zones+self.n_charging_stations]=np.ones(self.n_charging_stations)*self.scenario["outlet_per_stations"]
        if self.delay_queue=='single':
            if not self.av_trip_time:
                av_trips_time=np.mean(self.trips_time_per_zone())
                print("Average trip time (h): ", av_trips_time)
                self.delay_rate=1/av_trips_time
            else:
                self.delay_rate=1/(self.av_trip_time/60)
                print("Average given trip time (min): ", self.av_trip_time)
            service_rates=np.append(service_rates,self.delay_rate)
        elif self.delay_queue=='multiple':
            if not self.av_trip_time:
                trips_time=self.trips_time_per_zone()
                new_sr=1/(trips_time)
                self.delay_rate=np.mean(trips_time)
                print("Average trip time (h): ", self.delay_rate)
            else:
                self.delay_rate=1/(self.av_trip_time/60)
                print("Average given trip time (min): ", self.av_trip_time)
                new_sr=np.ones(self.n_zones)*(self.delay_rate) #all equal (as single delay queue)
            service_rates=np.append(service_rates,new_sr)
        return n_servers, service_rates
    
    def print_scenario(self):
        print("Number of zones: ", self.n_zones)
        print("Number of charging stations: ", self.n_charging_stations)
        if self.delay_queue=='single':
            print("Delay zone included with rate: ", self.delay_rate)
        elif self.delay_queue=='multiple':
            print("Delay zones included with average rate: ", self.delay_rate)
        print("Trips autonomy: ", np.round(self.trips_autonomy,2))
        print(f"Distance autonomy: {np.round(self.dist_autonomy,2)} km")
        if self.n_charging_stations!=0:
            print(f"Charging rates: {np.round(self.outlet_rate,2)}/h")
            print("Zones with charging stations", self.zones_with_charging_stations)
            print("Charging outlet per station: ", self.scenario["outlet_per_stations"])

    def change_scenario_param(self, param, value):
        for i in range(len(param)):
            self.scenario[param[i]]=value[i]
        with open(self.json_file, "w") as jsonFile:
            json.dump(self.scenario, jsonFile, indent=1)
        self.__init__(self.json_file)

    def average_dist(self):
        dist_matr=np.zeros((self.n_zones, self.n_zones))
        city_grid_3035=self.city_grid.to_crs(epsg=3035)
        city_centroids=city_grid_3035.centroid
        mat_id=0
        for zone in city_centroids:
            #print(city_centroids.distance(zone))
            dist_matr[mat_id,:]=city_centroids.distance(zone)
            mat_id+=1
        return dist_matr

    def average_speed(self):
        av_speed=np.divide(self.bookings_df.distance/1000,self.bookings_df.duration/3600)
        return np.mean(av_speed)

    def trips_time_per_zone(self):
        distance_matrix=self.average_dist()
        avg_dist_per_zone=np.zeros(self.n_zones)
        for zone in range(self.n_zones):
            avg_dist_per_zone[zone]=np.sum(np.multiply(self.OD_matrix[zone,:],distance_matrix[zone,:]))/1000
        #print(avg_dist_per_zone)
        av_speed=self.average_speed()
        av_trips_time=avg_dist_per_zone/av_speed
        #print(av_trips_time*60)
        #inv_zones_id_dict = {v: k for k, v in self.zones_id_dict.items()}
        #print(av_trips_time[inv_zones_id_dict[200]]*60)
        return av_trips_time

    def find_neighbors(self, grid, radius, row_number, column_number):
        return [[grid[i][j] if  i >= 0 and i < len(grid) and j >= 0 and j < len(grid[0]) else 0 for j in range(column_number-1-radius, column_number+radius)] for i in range(row_number-1-radius, row_number+radius)]