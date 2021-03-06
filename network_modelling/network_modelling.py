from city_scenario.city_scenario import Scenario

import numpy as np
import pandas as pd

class Network():
    def __init__(self, city_scenario):
        self.city_scenario=city_scenario
        self.n_zones=self.city_scenario.n_zones
        self.n_charging_stations=self.city_scenario.n_charging_stations
        self.n_vehicles=self.city_scenario.n_vehicles
        self.service_rates=self.city_scenario.service_rates
        self.n_servers=self.city_scenario.n_servers
        self.OD_matrix=self.city_scenario.OD_matrix
        self.trips_autonomy=self.city_scenario.trips_autonomy
        self.city_grid=self.city_scenario.city_grid
        self.zones_id_dict=self.city_scenario.zones_id_dict
        if self.n_charging_stations!=0:
            self.zones_with_charging_stations=self.city_scenario.zones_with_charging_stations
            self.charging_policy=self.city_scenario.charging_policy
            self.reloc_after_charging=self.city_scenario.reloc_after_charging
            self.n_top_zones=self.city_scenario.n_top_zones

        self.city_grid_data, self.city_grid_CS_data, self.aug_OD_matrix, self.ov_throughput=self.steady_state_analysis()
        if not self.city_grid_data.empty:
            if self.n_charging_stations>0:
                self.waiting_p=self.compute_waiting_probability()
            self.av_ud=np.sum(np.multiply(np.array(self.city_grid_data['un_demand']),(np.array(self.city_grid_data['service_rates']/np.sum(np.array(self.city_grid_data['service_rates']))))))

    def steady_state_analysis(self):
        """
        if self.charging_policy=='opportunistic':
            flows, aug_OD_matrix=self.total_fluxes()
        else:
        """
        if self.city_scenario.delay_queue=="multiple":
            flows, aug_OD_matrix=self.fluxes_with_reloc4charg_del()
        else:
            flows, aug_OD_matrix=self.fluxes_with_reloc4charg()
        if flows.size!=0:
            #print("F sum: ", np.sum(flows))
            ## MS_MVA - Multi-Server Mean Value Analysis of the network
            if self.city_scenario.delay_queue=="single":
                self.av_vehicles, self.av_delay, ov_throughput=self.MS_MVA_inf(flows, self.n_vehicles)
                print("Av vehicles in D: ", self.av_vehicles[self.n_zones+self.n_charging_stations])
                print("Av time in D: ", self.av_delay[self.n_zones+self.n_charging_stations])
            elif self.city_scenario.delay_queue=="multiple":
                self.av_vehicles, self.av_delay, ov_throughput=self.MS_MVA_inf_mult(flows, self.n_vehicles)
                print("Av vehicles in Ds:", np.sum(self.av_vehicles[self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations]))
                print("Av time in Ds:", np.sum(self.av_delay[self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations])/self.n_zones)
            else:
                self.av_vehicles, self.av_delay, ov_throughput=self.MS_MVA(flows, self.n_vehicles)
                print("NO DELAY")
            throughput_vector=ov_throughput*flows
            #print("TOTOT: ", np.sum(throughput_vector))
            #print("MVA OV:", ov_throughput)
            zone_throughput=np.sum(throughput_vector[0:self.n_zones])
            #print("ov throughput zones+CS only: ", zone_throughput)
            av_waiting=self.av_delay-(1/self.service_rates)
            av_servers_occupancy=np.multiply((1/self.service_rates),flows*ov_throughput)
            #compute utilization vector (rho) with computed flows and service rates
            rho=np.divide(throughput_vector[0:self.n_zones+self.n_charging_stations],np.multiply(self.service_rates[0:self.n_zones+self.n_charging_stations],self.n_servers))
            #Compute unsatisfied demand and mobility requests for mobility zones only
            unsatisfied_demand_per_zone=((1-rho[0:self.n_zones]))
            lost_requests_per_zone=np.round(np.multiply(unsatisfied_demand_per_zone,self.service_rates[0:self.n_zones]),4)
            if self.n_charging_stations>0:
                zones_with_charging_stations_ID=[]
                for zone in self.zones_with_charging_stations:
                    zones_with_charging_stations_ID.append(self.zones_id_dict[zone])  
            #build df with statistics over city grid
            city_grid_data=self.city_grid.copy()
            city_grid_data=city_grid_data.sort_values(by=['zone_id'])
            city_grid_data['service_rates']=self.service_rates[0:self.n_zones]
            city_grid_data['flows']=flows[0:self.n_zones]
            city_grid_data['throughput']=throughput_vector[0:self.n_zones]
            city_grid_data['av_vehicles']=self.av_vehicles[0:self.n_zones]
            city_grid_data['av_delay']=self.av_delay[0:self.n_zones]
            city_grid_data['utilization']=rho[0:self.n_zones]
            city_grid_data['un_demand']=unsatisfied_demand_per_zone[0:self.n_zones]*100
            city_grid_data['lost_req']=lost_requests_per_zone[0:self.n_zones]
            city_grid_data['av_server_occupancy']=av_servers_occupancy[0:self.n_zones]
            if self.n_charging_stations>0:
                city_grid_CS_data=self.city_grid.loc[self.city_grid['zone_id'].isin(zones_with_charging_stations_ID)]
                city_grid_CS_data=city_grid_CS_data.sort_values(by=['zone_id'])
                city_grid_CS_data['service_rates']=self.service_rates[self.n_zones:self.n_zones+self.n_charging_stations]
                city_grid_CS_data['flows']=flows[self.n_zones:self.n_zones+self.n_charging_stations]
                city_grid_CS_data['throughput']=throughput_vector[self.n_zones:self.n_zones+self.n_charging_stations]
                city_grid_CS_data['av_delay']=self.av_delay[self.n_zones:self.n_zones+self.n_charging_stations]
                city_grid_CS_data['av_vehicles']=self.av_vehicles[self.n_zones:self.n_zones+self.n_charging_stations]
                city_grid_CS_data['av_waiting']=av_waiting[self.n_zones:self.n_zones+self.n_charging_stations]
                city_grid_CS_data['utilization']=rho[self.n_zones:self.n_zones+self.n_charging_stations]
                city_grid_CS_data['av_server_occupancy']=av_servers_occupancy[self.n_zones:self.n_zones+self.n_charging_stations]
            else:
                city_grid_CS_data=pd.DataFrame()
        else:
            city_grid_data=pd.DataFrame()
            city_grid_CS_data=pd.DataFrame()
            zone_throughput=None

        return city_grid_data, city_grid_CS_data, aug_OD_matrix, zone_throughput

    def MS_MVA(self, flows, n_vehicles):
        n_queues=flows.size
        #n_queues=self.n_zones+self.n_charging_stations
        #Multi servers Mean Value Analysis (MS-MVA)
        average_vehicles=np.zeros((n_queues,n_vehicles+1)) #average number of vehicles per zone
        average_waiting=np.zeros((n_queues,n_vehicles+1)) #average "waiting time" of vehicles per zone
        max_ns=int(np.max(self.n_servers))
        p=np.zeros((n_queues,max_ns,n_vehicles+1))
        p[:,0,0]=1
        for m in range(1,n_vehicles+1):
            for n in range(n_queues):
                ns=int(self.n_servers[n])
                correction_factor=0
                for j in range(1,ns):
                    correction_factor+=(ns-j)*p[n,j-1,m-1]
                average_waiting[n,m]=(1+average_vehicles[n,m-1]+correction_factor)/(self.service_rates[n]*ns)
            overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
            average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
            for n in range (n_queues):
                ns=int(self.n_servers[n])
                su=0
                for j in range(1,ns):
                    p[n,j,m]=(1/j)*(flows[n]/(self.service_rates[n]*ns))*overall_throughput*p[n,j-1,m-1]
                    su+=(ns-j)*p[n,j,m]
                p[n,0,m]=1-(1/ns)*(flows[n]/(self.service_rates[n]*ns)*overall_throughput+su)
        return average_vehicles[:,-1], average_waiting[:,-1], overall_throughput

    def find_closest_CS(self):
        grid_with_CS=[self.zones_id_dict[i] for i in self.zones_with_charging_stations]
        grid_with_CS_geo=self.city_grid.loc[self.city_grid.index.isin(grid_with_CS)]['geometry']
        grid_with_CS_geo=grid_with_CS_geo.to_crs(epsg=3035)
        city_grid=self.city_grid.to_crs(epsg=3035)
        #grid_with_CS_geo=grid_with_CS_geo.centroid
        nearest_CS=[]
        for zone in city_grid.geometry:
            nearest_CS.append(grid_with_CS_geo.distance(zone.centroid).sort_values().index[0])
        return nearest_CS

    def compute_waiting_probability(self):
        rho=np.array(self.city_grid_CS_data['utilization'])
        throughput_vector=np.array(self.city_grid_CS_data['throughput'])
        #PROBABILITY TO WAIT IN CS (Erlang-C)
        waiting_p=np.zeros(self.n_charging_stations)
        for i in range(self.n_charging_stations):
            num=1/(np.math.factorial(self.n_servers[self.n_zones+i]))*(throughput_vector[i]/self.service_rates[self.n_zones+i])**self.n_servers[self.n_zones+i]*(1/(1-rho[i]))
            den=0
            for m in range(int(self.n_servers[self.n_zones+i])):
                den+=1/(np.math.factorial(m))*(throughput_vector[i]/self.service_rates[self.n_zones+i])**m
            den+=1/(np.math.factorial(self.n_servers[self.n_zones+i]))*(throughput_vector[i]/self.service_rates[self.n_zones+i])**self.n_servers[self.n_zones+i]*(1/(1-rho[i]))
            waiting_p[i]=num/den
        self.city_grid_CS_data['waiting_p']=waiting_p
        return waiting_p

    def print_stat(self):
        print("Number of vehicles: ", self.n_vehicles)
        print("Charging policy: ", self.charging_policy)
        print("\nRelative flows per zone:\n", self.city_grid_data['flows'])
        print("Relative flows in CS:\n", self.city_grid_CS_data['flows'])
        print("\nAverage # vehicles per zone:\n", np.round(self.city_grid_data['av_vehicles'],3))
        print("\nAverage # vehicles in CS:\n", np.round(self.city_grid_CS_data['av_vehicles'],3))
        #print("Average vehicles idle time vector: ", np.round(av_delay,3))
        #print('Average vehicles "waiting" time vector: ', np.round(av_waiting,3))
        print("Overall throughput (constant for the flow calculation): ", np.round(self.ov_throughput,4))
        print("Throughputs per zone:\n", np.round(self.city_grid_data['throughput'],3))
        print("Throughputs in CS:\n", np.round(self.city_grid_CS_data['throughput'],3))
        print("Utilization per zone:\n", np.round(self.city_grid_data['utilization'],3))
        print("Utilization per CS:\n", np.round(self.city_grid_CS_data['utilization'],3))
        print(f"Waiting probability in CS:\n {np.round(self.city_grid_CS_data['waiting_p']*100,3)}%")

    def MS_MVA_inf(self, flows, n_vehicles):
        n_queues=flows.size+1
        flows=np.append(flows,np.sum(flows[0:self.n_zones]))
        #n_queues=self.n_zones+self.n_charging_stations
        #Multi servers Mean Value Analysis (MS-MVA)
        average_vehicles=np.zeros((n_queues,n_vehicles+1)) #average number of vehicles per zone
        average_waiting=np.zeros((n_queues,n_vehicles+1)) #average "waiting time" of vehicles per zone
        max_ns=int(np.max(self.n_servers))
        p=np.zeros((n_queues,max_ns,n_vehicles+1))
        p[:,0,0]=1
        for m in range(1,n_vehicles+1):
            for n in range(n_queues-1):
                ns=int(self.n_servers[n])
                correction_factor=0
                for j in range(1,ns):
                    correction_factor+=(ns-j)*p[n,j-1,m-1]
                average_waiting[n,m]=(1+average_vehicles[n,m-1]+correction_factor)/(self.service_rates[n]*ns)
            average_waiting[n_queues-1,m]=1/self.service_rates[n_queues-1] #of delay queue
            overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
            average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
            for n in range (n_queues-1):
                ns=int(self.n_servers[n])
                su=0
                for j in range(1,ns):
                    p[n,j,m]=(1/j)*(flows[n]/(self.service_rates[n]*ns))*overall_throughput*p[n,j-1,m-1]
                    su+=(ns-j)*p[n,j,m]
                p[n,0,m]=1-(1/ns)*(flows[n]/(self.service_rates[n]*ns)*overall_throughput+su)
            p[n_queues-1,0,m]=1 #delay queue
        return average_vehicles[:,-1], average_waiting[:,-1], overall_throughput

    def find_top_demand_zones(self, n_top_zones):
        zone_rates=self.service_rates[0:self.n_zones]
        top_zones_idx=list(np.argsort(-zone_rates)[0:n_top_zones])
        return top_zones_idx
    
    def fluxes_with_reloc4charg(self):
        aug_matrix=np.zeros((self.n_zones+self.n_charging_stations,self.n_zones+self.n_charging_stations))
        aug_matrix[0:self.n_zones,0:self.n_zones]=self.OD_matrix
        zones_list=list(np.arange(0,self.n_zones))
        
        lambda_vec=np.random.rand(self.n_charging_stations+self.n_zones)
        lambda_vec=lambda_vec/(np.sum(lambda_vec))
        num_it=0
        k=1/self.trips_autonomy
        if self.n_charging_stations>0:
            zones_without_CS=[zone for zone in zones_list if zone not in self.zones_with_charging_stations]
            #redirect flow from each zone to CS
            if self.charging_policy=='closest_CS':
                closest_CS_ids=self.find_closest_CS()
                inv_zones_id_dict = {v: k for k, v in self.zones_id_dict.items()}
                closest_CS_od=[inv_zones_id_dict[i] for i in closest_CS_ids]
                self.zones_with_charging_stations=np.array(self.zones_with_charging_stations)
                for i in range(self.n_zones):
                    aug_matrix[0:self.n_zones,self.n_zones+(np.where(self.zones_with_charging_stations==closest_CS_od[i])[0].item())]+=self.OD_matrix[0:self.n_zones,i]*k
            elif self.charging_policy=='uniform': #uniform relocation
                for i in range(len(self.zones_with_charging_stations)):
                    #correct OD matrix with k
                    #fluxes directed to the zone with CS inside
                    aug_matrix[0:self.n_zones,self.n_zones+i]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*k
                    for j in range(len(zones_without_CS)):
                        #fluxes directed to zones WITHOUT CS inside
                        aug_matrix[0:self.n_zones,self.n_zones+i]=aug_matrix[0:self.n_zones,self.n_zones+i]+self.OD_matrix[0:self.n_zones,zones_without_CS[j]]*(k/self.n_charging_stations)
            #for both each mobility zone times (1-k)
            aug_matrix[0:self.n_zones,0:self.n_zones]=self.OD_matrix[0:self.n_zones,0:self.n_zones]*(1-k)
            if self.reloc_after_charging=='highest_demand':
                top_zones=self.find_top_demand_zones(self.n_top_zones)
                for j in range(self.n_charging_stations):
                    aug_matrix[self.n_zones+j][top_zones]=1/self.n_top_zones
            elif self.reloc_after_charging=='uniform':
                for j in range(self.n_charging_stations):
                    aug_matrix[self.n_zones+j][0:self.n_zones]=1/self.n_zones
            elif self.reloc_after_charging=='probabilistic':
                norm_rates=self.service_rates[0:self.n_zones]
                norm_rates=norm_rates/np.sum(norm_rates)
                for j in range(self.n_charging_stations):
                    aug_matrix[self.n_zones+j,0:self.n_zones]=norm_rates
            else:
                #define only element in the CS row equal to one corresponding to its zone
                for j in range(self.n_charging_stations):
                    aug_matrix[self.n_zones+j][self.zones_with_charging_stations[j]]=1
        #ITERATE TO FIND FLOWS AT STEADY STATE
        iterate=True
        while iterate:
            flows=np.dot(lambda_vec,aug_matrix)#compute vector of flows
            if self.n_charging_stations>0:
                if self.charging_policy=='opportunistic':
                    tot_CS_flows=0
                    for i in self.zones_with_charging_stations:
                        tot_CS_flows+=flows[i]
                    tot_flows=np.sum(flows)
                    k=(1/self.trips_autonomy)/(tot_CS_flows/tot_flows)
                    if k>1:
                        print("K theory: ",k)
                        k=1
                        print("K clipped")
                    #COMPUTE CHARGING FLOWS FOR EACH ZONE WITH CURRENT NETWORK FLOWS 
                    for i in range(len(self.zones_with_charging_stations)):
                        #correct OD matrix with k
                        aug_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*(1-k)
                        aug_matrix[0:self.n_zones,self.n_zones+i]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*k
                    aug_matrix=aug_matrix/aug_matrix.sum(axis=1)[:,None]    
            if num_it<50:
                if (abs(flows-lambda_vec)>10e-4).any(): #check on convergence
                    lambda_vec=flows
                    num_it+=1
                else:
                    iterate=False
                    flows=np.array([max(0,i) for i in flows])
            else:
                if num_it <10000:
                    if (abs(flows-lambda_vec)>10e-3).any(): #check on convergence
                        lambda_vec=flows
                        num_it+=1
                    else:
                        iterate=False
                        flows=np.array([max(0,i) for i in flows])
                else:
                    iterate=False
                    flows=np.array([])
        return flows, aug_matrix

    def total_fluxes(self):
        aug_matrix=np.zeros((self.n_zones+self.n_charging_stations,self.n_zones+self.n_charging_stations))
        aug_matrix[0:self.n_zones,0:self.n_zones]=self.OD_matrix
        #define only element in the CS row equal to one corresponding to its zone
        for j in range(self.n_charging_stations):
            aug_matrix[self.n_zones+j][self.zones_with_charging_stations[j]]=1
        lambda_vec=np.random.rand(self.n_charging_stations+self.n_zones)
        lambda_vec=lambda_vec/(np.sum(lambda_vec))
        num_it=0
        k=1/self.trips_autonomy
        #ITERATE TO FIND FLOWS AT STEADY STATE
        iterate=True
        while iterate:
            flows=np.dot(lambda_vec,aug_matrix)#compute vector of flows
            tot_CS_flows=0
            for i in self.zones_with_charging_stations:
                tot_CS_flows+=flows[i]
            tot_flows=np.sum(flows)
            k=(1/self.trips_autonomy)/(tot_CS_flows/tot_flows)
            if k>=1:
                k=0.99
                "K clipped"
            if self.n_charging_stations>0:
                #COMPUTE CHARGING FLOWS FOR EACH ZONE WITH CURRENT NETWORK FLOWS 
                for i in range(len(self.zones_with_charging_stations)):
                    #correct OD matrix with k
                    aug_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*(1-k)
                    aug_matrix[0:self.n_zones,self.n_zones+i]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*k
            aug_matrix=aug_matrix/aug_matrix.sum(axis=1)[:,None]    
            if num_it<50:
                if (abs(flows-lambda_vec)>10e-4).any(): #check on convergence
                    lambda_vec=flows
                    num_it+=1
                else:
                    iterate=False
                    flows=np.array([max(0,i) for i in flows])
            else:
                if num_it <10000:
                    if (abs(flows-lambda_vec)>10e-3).any(): #check on convergence
                        lambda_vec=flows
                        num_it+=1
                    else:
                        iterate=False
                        flows=np.array([max(0,i) for i in flows])
                else:
                    iterate=False
                    flows=np.array([]) 
        return flows, aug_matrix

    def fluxes_with_reloc4charg_old(self):
        aug_matrix=np.zeros((self.n_zones+self.n_charging_stations,self.n_zones+self.n_charging_stations))
        aug_matrix[0:self.n_zones,0:self.n_zones]=self.OD_matrix
        zones_list=list(np.arange(0,self.n_zones))
        zones_without_CS=[zone for zone in zones_list if zone not in self.zones_with_charging_stations]
        #define only element in the CS row equal to one corresponding to its zone
        for j in range(self.n_charging_stations):
            aug_matrix[self.n_zones+j][self.zones_with_charging_stations[j]]=1
        lambda_vec=np.random.rand(self.n_charging_stations+self.n_zones)
        lambda_vec=lambda_vec/(np.sum(lambda_vec))
        num_it=0
        k=1/self.trips_autonomy
        if self.n_charging_stations>0:
            #redirect flow from each zone to CS
            if self.charging_policy=='closest_CS':
                closest_CS_ids=self.find_closest_CS()
                inv_zones_id_dict = {v: k for k, v in self.zones_id_dict.items()}
                closest_CS_od=[inv_zones_id_dict[i] for i in closest_CS_ids]
                self.zones_with_charging_stations=np.array(self.zones_with_charging_stations)
                for i in range(self.n_zones):
                    aug_matrix[0:self.n_zones,self.n_zones+(np.where(self.zones_with_charging_stations==closest_CS_od[i])[0].item())]+=self.OD_matrix[0:self.n_zones,i]*k
            else: #uniform relocation
                for i in range(len(self.zones_with_charging_stations)):
                    #correct OD matrix with k
                    #fluxes directed to the zone with CS inside
                    aug_matrix[0:self.n_zones,self.n_zones+i]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*k
                    for j in range(len(zones_without_CS)):
                        #fluxes directed to zones WITHOUT CS inside
                        aug_matrix[0:self.n_zones,self.n_zones+i]=aug_matrix[0:self.n_zones,self.n_zones+i]+self.OD_matrix[0:self.n_zones,zones_without_CS[j]]*(k/self.n_charging_stations)
            #for both each mobility zone times (1-k)
            aug_matrix[0:self.n_zones,0:self.n_zones]=self.OD_matrix[0:self.n_zones,0:self.n_zones]*(1-k)        
        #ITERATE TO FIND FLOWS AT STEADY STATE
        iterate=True
        while iterate:
            flows=np.dot(lambda_vec,aug_matrix)#compute vector of flows
            if num_it<50:
                if (abs(flows-lambda_vec)>10e-4).any(): #check on convergence
                    lambda_vec=flows
                    num_it+=1
                else:
                    iterate=False
                    flows=np.array([max(0,i) for i in flows])
            else:
                if num_it <10000:
                    if (abs(flows-lambda_vec)>10e-3).any(): #check on convergence
                        lambda_vec=flows
                        num_it+=1
                    else:
                        iterate=False
                        flows=np.array([max(0,i) for i in flows])
                else:
                    iterate=False
                    flows=np.array([])
        return flows, aug_matrix

    def fluxes_with_reloc4charg_del(self):
        """
        np.random.seed(32)
        self.n_zones=4
        self.n_charging_stations=2
        self.OD_matrix=np.random.rand(self.n_zones,self.n_zones)
        self.OD_matrix=self.OD_matrix/self.OD_matrix.sum(axis=1)[:,None]
        self.zones_with_charging_stations=[0,2]
        print("OD: ",self.OD_matrix)
        """
        aug_matrix=np.zeros((self.n_zones+self.n_charging_stations,self.n_zones+self.n_charging_stations))
        aug_matrix[0:self.n_zones,0:self.n_zones]=self.OD_matrix
        zones_list=list(np.arange(0,self.n_zones))
        zones_without_CS=[zone for zone in zones_list if zone not in self.zones_with_charging_stations]
        lambda_vec=np.random.rand(self.n_charging_stations+self.n_zones*2)
        lambda_vec=lambda_vec/(np.sum(lambda_vec))
        num_it=0
        k=1/self.trips_autonomy
        if self.n_charging_stations>0:
            #redirect flow from each zone to CS
            if self.charging_policy=='closest_CS':
                closest_CS_ids=self.find_closest_CS()
                inv_zones_id_dict = {v: k for k, v in self.zones_id_dict.items()}
                closest_CS_od=[inv_zones_id_dict[i] for i in closest_CS_ids]
                self.zones_with_charging_stations=np.array(self.zones_with_charging_stations)
                for i in range(self.n_zones):
                    aug_matrix[0:self.n_zones,self.n_zones+(np.where(self.zones_with_charging_stations==closest_CS_od[i])[0].item())]+=self.OD_matrix[0:self.n_zones,i]*k
            elif self.charging_policy=='uniform': #uniform relocation
                for i in range(len(self.zones_with_charging_stations)):
                    #correct OD matrix with k
                    #fluxes directed to the zone with CS inside
                    aug_matrix[0:self.n_zones,self.n_zones+i]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*k
                    for j in range(len(zones_without_CS)):
                        #fluxes directed to zones WITHOUT CS inside
                        aug_matrix[0:self.n_zones,self.n_zones+i]=aug_matrix[0:self.n_zones,self.n_zones+i]+self.OD_matrix[0:self.n_zones,zones_without_CS[j]]*(k/self.n_charging_stations)
            #for both each mobility zone times (1-k)
            aug_matrix[0:self.n_zones,0:self.n_zones]=self.OD_matrix[0:self.n_zones,0:self.n_zones]*(1-k)
            if self.reloc_after_charging=='highest_demand':
                top_zones=self.find_top_demand_zones(self.n_top_zones)
                for j in range(self.n_charging_stations):
                    aug_matrix[self.n_zones+j][top_zones]=1/self.n_top_zones
            elif self.reloc_after_charging=='uniform':
                for j in range(self.n_charging_stations):
                    aug_matrix[self.n_zones+j][0:self.n_zones]=1/self.n_zones
            else:
                #define only element in the CS row equal to one corresponding to its zone
                for j in range(self.n_charging_stations):
                    aug_matrix[self.n_zones+j][self.zones_with_charging_stations[j]]=1
        #print("AUG:\n", np.round(aug_matrix,2))
        ##ONE DELAY ZONE for each departure zone
        delay_mat=np.zeros((self.n_zones*2+self.n_charging_stations,self.n_zones*2+self.n_charging_stations))
        delay_mat[0:self.n_zones,self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations]=np.identity(self.n_zones)
        delay_mat[self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations,0:self.n_zones]=aug_matrix[0:self.n_zones,0:self.n_zones]
        delay_mat[self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations,self.n_zones:self.n_zones+self.n_charging_stations]=aug_matrix[0:self.n_zones,self.n_zones:self.n_zones+self.n_charging_stations]
        
        if self.reloc_after_charging=='highest_demand' or self.reloc_after_charging=='uniform':
            #print("UNIFORM")
            delay_mat[self.n_zones:self.n_zones+self.n_charging_stations,self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations]=aug_matrix[self.n_zones:self.n_zones+self.n_charging_stations,0:self.n_zones]
            for j in range(len(self.zones_with_charging_stations)):
                to_same_zone=delay_mat[self.n_zones+j,self.n_zones+self.n_charging_stations+self.zones_with_charging_stations[j]]
                delay_mat[self.n_zones+j,self.n_zones+self.n_charging_stations+self.zones_with_charging_stations[j]]=0
                delay_mat[self.n_zones+j,self.zones_with_charging_stations[j]]=to_same_zone
        else:
            delay_mat[self.n_zones:self.n_zones+self.n_charging_stations,0:self.n_zones]=aug_matrix[self.n_zones:self.n_zones+self.n_charging_stations,0:self.n_zones]
        #print("DELAY:\n", np.round(delay_mat,2))
        #ITERATE TO FIND FLOWS AT STEADY STATE
        iterate=True
        while iterate:
            flows=np.dot(lambda_vec,delay_mat)#compute vector of flows
            #print(np.round(flows,2))
            #print("DEL:\n", np.round(delay_mat,2))
            if self.charging_policy=='opportunistic':
                tot_CS_flows=0
                for i in self.zones_with_charging_stations:
                    tot_CS_flows+=flows[i]
                tot_flows=np.sum(flows)
                k=(1/self.trips_autonomy)/(tot_CS_flows/tot_flows)
                if k>=1:
                    k=0.99
                    "K clipped"
                if self.n_charging_stations>0:
                    #COMPUTE CHARGING FLOWS FOR EACH ZONE WITH CURRENT NETWORK FLOWS 
                    for i in range(len(self.zones_with_charging_stations)):
                        #correct OD matrix with k
                        delay_mat[self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations,self.zones_with_charging_stations[i]]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*(1-k)
                        delay_mat[self.n_zones+self.n_charging_stations:self.n_zones*2+self.n_charging_stations,self.n_zones+i]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*k
                        #aug_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*(1-k)
                        #aug_matrix[0:self.n_zones,self.n_zones+i]=self.OD_matrix[0:self.n_zones,self.zones_with_charging_stations[i]]*k
                delay_mat=delay_mat/delay_mat.sum(axis=1)[:,None]
                #aug_matrix=aug_matrix/aug_matrix.sum(axis=1)[:,None]    
            if num_it<50:
                if (abs(flows-lambda_vec)>10e-4).any(): #check on convergence
                    lambda_vec=flows
                    num_it+=1
                else:
                    iterate=False
                    flows=np.array([max(0,i) for i in flows])
            else:
                if num_it <10000:
                    if (abs(flows-lambda_vec)>10e-3).any(): #check on convergence
                        lambda_vec=flows
                        num_it+=1
                    else:
                        iterate=False
                        flows=np.array([max(0,i) for i in flows])
                else:
                    iterate=False
                    flows=np.array([])
        return flows, delay_mat

    def MS_MVA_inf_mult(self, flows, n_vehicles):
        n_queues=flows.size
        #flows=np.append(flows,np.sum(flows[0:self.n_zones]))
        #n_queues=self.n_zones+self.n_charging_stations
        #Multi servers Mean Value Analysis (MS-MVA)
        average_vehicles=np.zeros((n_queues,n_vehicles+1)) #average number of vehicles per zone
        average_waiting=np.zeros((n_queues,n_vehicles+1)) #average "waiting time" of vehicles per zone
        max_ns=int(np.max(self.n_servers))
        p=np.zeros((n_queues,max_ns,n_vehicles+1))
        p[:,0,0]=1
        for m in range(1,n_vehicles+1):
            for n in range(self.n_zones+self.n_charging_stations):
                ns=int(self.n_servers[n])
                correction_factor=0
                for j in range(1,ns):
                    correction_factor+=(ns-j)*p[n,j-1,m-1]
                average_waiting[n,m]=(1+average_vehicles[n,m-1]+correction_factor)/(self.service_rates[n]*ns)
            average_waiting[self.n_zones+self.n_charging_stations:n_queues-1,m]=1/self.service_rates[n_queues-1] #of delay queues
            overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
            average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
            for n in range (self.n_zones+self.n_charging_stations):
                ns=int(self.n_servers[n])
                su=0
                for j in range(1,ns):
                    p[n,j,m]=(1/j)*(flows[n]/(self.service_rates[n]*ns))*overall_throughput*p[n,j-1,m-1]
                    su+=(ns-j)*p[n,j,m]
                p[n,0,m]=1-(1/ns)*(flows[n]/(self.service_rates[n]*ns)*overall_throughput+su)
            p[self.n_zones+self.n_charging_stations:n_queues-1,0,m]=1 #delay queue
        return average_vehicles[:,-1], average_waiting[:,-1], overall_throughput

    