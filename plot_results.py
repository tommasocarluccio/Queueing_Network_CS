import numpy as np
import matplotlib.pyplot as plt
from network_modelling.network_modelling import Network

class Results():
    def __init__(self, city_scenario, network_model, case):
        self.city_scenario=city_scenario
        self.network_model=network_model

        self.city_grid_data=self.network_model.city_grid_data
        self.city_grid_CS_data=self.network_model.city_grid_CS_data
        self.ov_throughput=self.network_model.ov_throughput
        self.aug_OD_matrix=self.network_model.aug_OD_matrix
        try:
            self.av_ud=self.network_model.av_ud
        except:
            pass
        self.n_zones=self.city_scenario.n_zones
        self.n_charging_stations=self.city_scenario.n_charging_stations
        self.n_vehicles=self.city_scenario.scenario["n_vehicles"]
        self.service_rates=self.city_scenario.service_rates
        self.n_servers=self.city_scenario.n_servers
        self.zones_id_dict=self.city_scenario.zones_id_dict
        self.zones_with_charging_stations=self.city_scenario.zones_with_charging_stations

        self.case=case

    def plot_data_on_grid(self):
        #fig1, (ax11,ax12,ax13)=plt.subplots(1,3)
        #self.city_grid_data.plot(color="white", edgecolor="black")
        #plot over city grid
        fig7, (ax11,ax12,ax13)=plt.subplots(1,3)
        fig7.set_size_inches(18.5, 10.5)
        fig7.text(0.5, 0.98, f"System throughput: {np.round(self.ov_throughput,2)}\nAverage unsatisfied demand: {np.round(self.av_ud,2)}%",horizontalalignment='center', verticalalignment='top')
        self.city_grid_CS_data=self.city_grid_CS_data.to_crs(epsg=3035)
        CS_centroids=self.city_grid_CS_data.centroid
        CS_centroids=CS_centroids.to_crs(epsg=4326)
        self.city_grid_data.plot(column="av_vehicles", ax=ax11, legend=True, legend_kwds={'label':'Average vehicles'},edgecolor="black")
        CS_centroids.plot(ax=ax11, marker='.', color="red", markersize=24)
        self.city_grid_data.plot(column="un_demand", ax=ax12, legend=True, legend_kwds={'label':'Unsatisfied demand [%]'},edgecolor="black")
        CS_centroids.plot(ax=ax12, marker='.', color="red", markersize=24)
        self.city_grid_data.plot(column="throughput", ax=ax13, legend=True, legend_kwds={'label':'Throughput'},edgecolor="black")
        CS_centroids.plot(ax=ax13, marker='.', color="red", markersize=24)
        ax11.set_axis_off()
        ax12.set_axis_off()
        ax13.set_axis_off()
        output_dir=f'img/{self.case}'
        self.mkdir_p(output_dir)
        plt.savefig(f'{output_dir}/map_plot.png', transparent=True, bbox_inches='tight')
        print("ov throughput: ", self.ov_throughput)
        print("avg unsatisfied demand: ", self.av_ud)
        #plt.show() 

    def plot_results(self, range_vehicles):
        output_dir=f'img/{self.case}'
        flows=np.array(self.city_grid_data['flows'])
        _aug_OD=self.aug_OD_matrix
        fig4, ax4 = plt.subplots()
        fig4.set_size_inches(18.5, 10.5)
        for _n_vehicles in range_vehicles:
            _av_vehicles, _av_delay, _ov_throughput=Network.MS_MVA(self, flows, _n_vehicles)
            _throughput_vector=_ov_throughput*flows
            _rho=np.round(np.divide(_throughput_vector,np.multiply(self.service_rates[0:self.n_zones],self.n_servers[0:self.n_zones])),4)
            _unsatisfied_demand_per_zone=((1-_rho))
            ax4.hist(_unsatisfied_demand_per_zone, bins=30, cumulative=True, density=True, histtype='step', label=f'n vehicles: {_n_vehicles}')
        ax4.set_title(f"Cumulative distribution of unsatisfied mobility demand per zones with {self.n_zones} zones")
        ax4.set_xlabel("Unsatisfied demand")
        ax4.grid()
        ax4.legend(loc='upper left')
        plt.savefig(f'{output_dir}/cum_dist_range.png', transparent=True, bbox_inches='tight')
        #plt.show()
        grid_with_CS=[self.zones_id_dict[i] for i in self.zones_with_charging_stations]
        cs_ticks=["z"+str(i) for i in grid_with_CS]
        
        fig6, (ax61,ax62)=plt.subplots(2,1, sharex=True)
        fig6.set_size_inches(18.5, 10.5)
        ax61.bar(np.arange(self.n_charging_stations),self.city_grid_CS_data['av_vehicles'], alpha=.5)
        ax62.bar(np.arange(self.n_charging_stations),self.city_grid_CS_data['waiting_p']*100, color='green')
        fig6.suptitle("Vehicles and waiting proability in charging stations")
        ax61.set_title("Average number of vehicles per charging station")
        ax62.set_title("Vehicles waiting probability per charging station")
        ax62.set_xlabel("Charging stations")
        ax61.set_ylabel("Vehicles")
        ax62.set_ylabel("Probability [%]")
        ax61.set_xticks(list(np.arange(self.n_charging_stations)))
        ax61.set_xticklabels(cs_ticks, rotation=45)
        ax62.set_xticks(list(np.arange(self.n_charging_stations)))
        ax62.set_xticklabels(cs_ticks, rotation=45)
        ax61.grid()
        ax62.grid()
        plt.savefig(f'{output_dir}/wait_p.png', transparent=True, bbox_inches='tight')
        #ax61.scatter(np.arange(n_charging_stations),n_servers[n_zones:n_zones+n_charging_stations], label=f'n servers', color='red',marker='.')
        #fig6.legend()
        #plt.show()

        #cumulative distribution histogram of unsatisfied demand
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(18.5, 10.5)
        ax3.hist(self.city_grid_data['un_demand'], bins=30, cumulative=True, density=True, histtype='stepfilled' )
        ax3.set_title(f"Cumulative distribution of unsatisfied mobility demand per zones {self.n_vehicles} vehicles")
        ax3.set_xlabel("Unsatisfied demand")
        ax3.grid()
        plt.savefig(f'{output_dir}/cum_dist.png', transparent=True, bbox_inches='tight')
      
    def plot_bar_per_zone(self):
        output_dir=f'img/{self.case}'
        zone_axis=[self.zones_id_dict[i] for i in np.arange(self.n_zones)]
        grid_with_CS=[self.zones_id_dict[i] for i in self.zones_with_charging_stations]
        cs_ticks=["z"+str(i) for i in grid_with_CS]
        #AVG VEHICLES
        fig11, (ax11, ax21) = plt.subplots(1,2)
        fig11.set_size_inches(18.5, 10.5)
        ax11.bar(zone_axis,self.city_grid_data['av_vehicles']) 
        #ax21.bar(np.arange(n_charging_stations),n_servers[n_zones:n_zones+n_charging_stations],label='number of servers', color='red', alpha=0.7)
        ax21.bar(np.arange(self.n_charging_stations),self.city_grid_CS_data['av_vehicles'], label='number of vehicles', alpha=0.7)
        fig11.suptitle(f"Average vehicles per zones")
        ax11.set_xlabel("Zones")
        ax21.set_xlabel("Charging Stations")
        ax11.set_ylabel("Vehicles")
        ax21.set_xticks(list(np.arange(self.n_charging_stations)))
        ax21.set_xticklabels(cs_ticks, rotation=45)
        ax11.grid()
        ax21.grid()
        ax21.legend()
        plt.savefig(f'{output_dir}/bar_veh.png', transparent=True, bbox_inches='tight')
        #plt.show()
        #IDLE+WAITING TIME
        fig12, (ax12,ax22, ax32) = plt.subplots(1,3)
        fig12.set_size_inches(18.5, 10.5)
        ax12.bar(zone_axis,self.city_grid_data['av_delay'])
        ax22.bar(np.arange(self.n_charging_stations),self.city_grid_CS_data['av_delay'])
        ax32.bar(np.arange(self.n_charging_stations),self.city_grid_CS_data['av_waiting'])
        ax22.sharey(ax32)
        fig12.suptitle(f"Average vehicles delays per zones")
        ax12.set_title("Vehicles idle time")
        ax22.set_title("Total vehicles time in the queue")
        ax32.set_title("Vehicles waiting time")
        ax12.set_xlabel("Zones")
        ax22.set_xlabel("Charging Stations")
        ax32.set_xlabel("Charging Stations")
        ax12.set_ylabel("time [s]")
        ax22.set_ylabel("time [s]")
        ax32.set_ylabel("time [s]")
        ax22.set_xticks(list(np.arange(self.n_charging_stations)))
        ax22.set_xticklabels(cs_ticks)
        ax32.set_xticks(list(np.arange(self.n_charging_stations)))
        ax32.set_xticklabels(cs_ticks, rotation=45)
        ax12.grid()
        ax22.grid()
        ax32.grid()
        plt.savefig(f'{output_dir}/bar_del.png', transparent=True, bbox_inches='tight')
        #plt.show()
        #THROUGHPUT
        fig13, (ax13, ax23) = plt.subplots(1,2)
        fig13.set_size_inches(18.5, 10.5)
        ax13.bar(zone_axis,self.city_grid_data['throughput'])
        ax23.bar(np.arange(self.n_charging_stations), self.city_grid_CS_data['throughput'],color='red')
        fig13.suptitle(f"Vehicles througput per zones")
        ax13.set_xlabel("Zones")
        ax23.set_xlabel("Charging Stations")
        ax13.set_ylabel("Throughput")
        ax23.set_ylabel("Throughput")
        ax23.set_xticks(list(np.arange(self.n_charging_stations)))
        ax23.set_xticklabels(cs_ticks, rotation=45)
        ax13.grid()
        ax23.grid()
        plt.savefig(f'{output_dir}/bar_thr.png', transparent=True, bbox_inches='tight')
        #plt.show()
        #UTILIZATION
        fig14, (ax14, ax24) = plt.subplots(1,2, sharey=True)
        fig14.set_size_inches(18.5, 10.5)
        ax14.bar(zone_axis,self.city_grid_data['utilization']*100)
        ax24.bar(np.arange(self.n_charging_stations), self.city_grid_CS_data['utilization']*100, color='green')
        fig14.suptitle(f"Utilization per zones")
        ax14.set_xlabel("Zones")
        ax24.set_xlabel("Charging station")
        ax14.set_ylabel("Utilization [%]")
        ax24.set_xticks(list(np.arange(self.n_charging_stations)))
        ax24.set_xticklabels(cs_ticks, rotation=45)
        ax14.grid()
        ax24.grid()
        plt.savefig(f'{output_dir}/bar_ut.png', transparent=True, bbox_inches='tight')
        #plt.show()
        #UN DEMAND
        fig15, ax15 = plt.subplots()
        fig15.set_size_inches(18.5, 10.5)
        ax15.bar(zone_axis,self.city_grid_data['un_demand']*100)
        ax15.set_title(f"Unsatisfied mobility demand per zones")
        ax15.set_xlabel("Zones")
        ax15.set_ylabel("Unsatisfied demand [%]")
        ax15.grid()
        plt.savefig(f'{output_dir}/bar_und.png', transparent=True, bbox_inches='tight')
        #plt.show()
        #LOST REQ
        fig16, ax16 = plt.subplots()
        fig16.set_size_inches(18.5, 10.5)
        ax16.bar(zone_axis,self.city_grid_data['lost_req'])
        ax16.set_title(f"Lost mobility requests per zones")
        ax16.set_xlabel("Zones")
        ax16.set_ylabel("Lost requests")
        ax16.grid()
        plt.savefig(f'{output_dir}/bar_lost.png', transparent=True, bbox_inches='tight')
        #plt.show()
        #CS THR+ CS UT
        fig17, (ax17, ax27) = plt.subplots(1,2)
        avg_CS_utilization=self.city_grid_CS_data['utilization'].mean()*100
        avg_CS_throughput=self.city_grid_CS_data['throughput'].mean()
        fig17.set_size_inches(18.5, 10.5)
        ax17.bar(np.arange(self.n_charging_stations),self.city_grid_CS_data['throughput'], color='red')
        ax27.bar(np.arange(self.n_charging_stations), self.city_grid_CS_data['utilization']*100, color='green')
        fig17.suptitle(f"Throughput and utilization of charging stations")
        ax17.set_xlabel("Charging station")
        ax27.set_xlabel("Charging station")
        ax17.set_ylabel("Throughput")
        ax27.set_ylabel("Utilization [%]")
        ax17.set_xticks(list(np.arange(self.n_charging_stations)))
        ax27.set_xticks(list(np.arange(self.n_charging_stations)))
        ax17.set_xticklabels(cs_ticks, rotation=45)
        ax27.set_xticklabels(cs_ticks, rotation=45)
        ax17.set_title(f"Average CS throughput: {np.round(avg_CS_throughput,2)}")
        ax27.set_title(f"Average CS utilization: {np.round(avg_CS_utilization,2)}%")
        ax17.grid()
        ax27.grid()
        plt.savefig(f'{output_dir}/bar_CS_th_ut.png',transparent=True, bbox_inches='tight')

    def mkdir_p(self, mypath):
        from errno import EEXIST
        from os import makedirs,path
        try:
            makedirs(mypath)
        except OSError as exc: # Python >2.5
            if exc.errno == EEXIST and path.isdir(mypath):
                pass
            else: raise
