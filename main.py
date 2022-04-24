from cmath import nan
from locale import normalize
from operator import index
from turtle import color
import numpy as np
import scipy.linalg as la
import scipy.special
import itertools
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import geopandas as gpd
from haversine import haversine, Unit
from shapely.geometry import Point, LineString, Polygon
from sklearn import neighbors

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def build_OD(n_zones, cycles=False):
    matrix = np.random.rand(n_zones,n_zones)
    if not cycles:
        np.fill_diagonal(matrix,np.zeros(n_zones))
    matrix=matrix/matrix.sum(axis=1)[:,None]
    #print(matrix)
    #print("eigenvalues:", np.linalg.eig(matrix)[0])
    return np.round(matrix,4)

def build_distance_matrix(n_zones,n_charging_stations,zones_with_charging_stations):
    sub_matrix=np.random.uniform(low=1, high=10, size=(n_zones,n_zones))
    np.fill_diagonal(sub_matrix,np.zeros(n_zones))
    sub_matrix=(sub_matrix + sub_matrix.T)/2

    matrix=np.zeros((n_zones+n_charging_stations,n_zones+n_charging_stations))
    matrix[0:n_zones,0:n_zones]=sub_matrix
    #copy row and column of zone to line corresponding to CS in that zone
    for i in range(n_charging_stations):
        matrix[n_zones+i,:]=matrix[zones_with_charging_stations[i],:]
        matrix[:,n_zones+i]=matrix[:,zones_with_charging_stations[i]]
    
    return np.round(matrix,4)

def compute_charging_flows(n_zones, fluxes, OD_matrix, dist_matrix, dist_autonomy):
    dist_vector=[]
    #rel_flows=np.divide(fluxes,np.sum(fluxes))
    rel_flows=fluxes
    for i in range(n_zones):
        prob_dist=np.multiply(OD_matrix[:,i],dist_matrix[:,i])
        dist_vector.append(np.sum(np.multiply(prob_dist,rel_flows)))
    charging_flows=np.array(dist_vector)/dist_autonomy
    return charging_flows
    
def compute_fluxes(n_zones, OD_matrix):
    lambda_vec=np.ones((n_zones)) #initialize vector of flows
    num_it=0
    iterate=True
    while iterate:
        flows=np.dot(lambda_vec,OD_matrix) #compute vector of flows
        if (abs(flows-lambda_vec)>10e-4).any(): #check on convergence
            lambda_vec=flows
            num_it+=1
        else:
            iterate=False
    print("num it: ", num_it)
    return np.round(flows,4)

def compute_normalization_constant(n_zones, n_vehicles, rho):
    #Convolution algorithm for Normalization coefficient v2 (more efficient)
    g1=np.zeros(n_vehicles+1)
    g1[0]=1
    for m in range(0,n_zones):
        for n in range(1,n_vehicles+1):
            g1[n]=g1[n]+rho[m]*g1[n-1]
    #print("time: ", time.time()-t1)
    return g1[n_vehicles]

def compute_A(vehicles, service_rate, n_servers):
    A=1
    for j in range(1,vehicles+1):
        mu_j=min(j*service_rate,n_servers*service_rate)
        A*=(mu_j/service_rate)
    #print(f"ns:{n_servers}, A:{A}")
    return A

def compute_normalization_constant_MS(n_zones, n_vehicles, service_rates, rho, n_servers):
    g=np.zeros((n_vehicles+1,n_zones))
    g[0,:]=1
    for m in range(1,n_vehicles+1):
        A=compute_A(m,service_rates[0],n_servers[0])
        g[m,0]=(rho[0]**m)/A
    for n in range(n_zones):
        for m in range(1,n_vehicles+1):
            su=0
            for j in range(m+1):
                A=compute_A(j,service_rates[n],n_servers[n])
                su+=(rho[n]**j/A)*g[m-j,n-1]
            g[m,n]=su
    return g[n_vehicles,n_zones-1]
   
def MVA(n_zones, n_vehicles, service_rates, flows):
    #Mean Value Analysis (MVA)
    average_vehicles=np.zeros((n_zones,n_vehicles+1)) #average number of vehicles per zone
    average_waiting=np.zeros((n_zones,n_vehicles+1)) #average "waiting time" of vehicles per zone
    for m in range(1,n_vehicles+1):
        for n in range(n_zones):
            average_waiting[n,m]=(1+average_vehicles[n,m-1])/(service_rates[n])
        overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
        average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
    return average_vehicles[:,-1], average_waiting[:,-1], overall_throughput

def MS_MVA(n_zones, n_vehicles, service_rates, flows, n_servers):
    #Multi servers Mean Value Analysis (MS-MVA)
    average_vehicles=np.zeros((n_zones,n_vehicles+1)) #average number of vehicles per zone
    average_waiting=np.zeros((n_zones,n_vehicles+1)) #average "waiting time" of vehicles per zone
    max_ns=int(np.max(n_servers))
    p=np.zeros((n_zones,max_ns,n_vehicles+1))
    p[:,0,0]=1
    for m in range(1,n_vehicles+1):
        for n in range(n_zones):
            ns=int(n_servers[n])
            correction_factor=0
            for j in range(1,ns):
                correction_factor+=(ns-j)*p[n,j-1,m-1]
            average_waiting[n,m]=(1+average_vehicles[n,m-1]+correction_factor)/(service_rates[n]*ns)
        overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
        average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
        for n in range (n_zones):
            ns=int(n_servers[n])
            su=0
            for j in range(1,ns):
                p[n,j,m]=(1/j)*(flows[n]/(service_rates[n]*ns))*overall_throughput*p[n,j-1,m-1]
                su+=(ns-j)*p[n,j,m]
            p[n,0,m]=1-(1/ns)*(flows[n]/(service_rates[n]*ns)*overall_throughput+su)
    return average_vehicles[:,-1], average_waiting[:,-1], overall_throughput

def compute_generic_pi(n_zones, vehicles_vector, rho, normalization_constant, n_servers):
    pi=1
    for n in range(n_zones):
        if n_servers[n]>1:
            if vehicles_vector[n]<n_servers[n]:
                beta=rho[n]**(vehicles_vector[n])/(np.math.factorial(vehicles_vector[n]))
            else:
                beta=rho[n]**(vehicles_vector[n])/(np.math.factorial(n_servers[n])*(n_servers[n]**(vehicles_vector[n]-n_servers[n])))
        else:
            beta=rho[n]**(vehicles_vector[n])
        pi*=beta
    pi=pi/normalization_constant
    return pi

def find_states(array, n_vehicles, n_zones, level, states):
    if level == len(array) - 1:
        # parte per generare effettivamente lo stato e aggiungerlo alla lista
        array[-1] = n_vehicles - sum(array[:-1])
        states.append(array.copy())
    else:
        if level == 0:
            end_loop = n_vehicles + 1
        else:
            end_loop = n_vehicles - sum(array[:level]) + 1
        array[level] = 0
        while array[level] < end_loop:
            find_states(array, n_vehicles, n_zones, level + 1, states)
            array[level] += 1

def compute_pi0_rec(states, rho, normalization_constant, service_rates):
    tot_pi0=0
    tot_requests_lost=0
    empty_queues=[]
    for state in states:
        if 0 in state:
            pi=np.round(np.prod(rho**(np.array(state)))/normalization_constant,4)
            requests_lost=np.round(np.sum(np.multiply(service_rates,pi)),4)
            empty_queues.append((state,pi,requests_lost))
            tot_requests_lost+=requests_lost
            tot_pi0+=pi
    return empty_queues, tot_pi0, tot_requests_lost

def compute_pi0_comb(n_zones, n_vehicles, rho, normalization_constant):
        tot_pi_0=0
        tot_requests_lost=0
        empty_queues=[]
        v_vector=range(1,n_vehicles+1)
        #fixing n queues with zero vehicles and find all possible arangements of n_vehicles in n_zones-n 
        for num_zeros in range(1,n_zones):
            for vehicle_per_zone in itertools.product(v_vector, repeat=n_zones-num_zeros):
                if np.sum(vehicle_per_zone)==n_vehicles: 
                    for partial_rho in itertools.combinations(rho,n_zones-num_zeros): #possible rho ignoring zones with zero vehicles
                        pi=np.round(np.prod(partial_rho**(np.array(vehicle_per_zone)))/normalization_constant,4)
                        id=np.isin(rho,partial_rho) #indexes of zone with vehicles
                        partial_mu=[service_rates[i] for i in range(service_rates.size) if id[i]==False]
                        requests_lost=np.round((np.sum(partial_mu))*pi,4)
                        pos=0
                        v_complete=[]
                        for i in range(id.size):
                            if id[i]:
                                v_complete.append(vehicle_per_zone[pos])
                                pos+=1
                            else:
                                v_complete.append(0)
                        empty_queues.append((v_complete,pi,requests_lost))
                        tot_requests_lost+=requests_lost
                        tot_pi_0+=pi
        #print("empty queues: ", empty_queues)
        #print("Tot pi0: ", np.round(tot_pi_0,4))
        return empty_queues, np.round(tot_pi_0,4), np.round(tot_requests_lost,4)

def plot_pi0(n_zones, n_vehicles_max, rho):
        tot_pi0_vector=[]
        vehicles_range=range(1,n_vehicles_max)
        for n_vehicles in vehicles_range:
            normalization_constant=compute_normalization_constant(n_zones, n_vehicles, rho)
            tot_pi0_vector.append(compute_pi0_comb(n_zones, n_vehicles, rho, normalization_constant))
        
        fig, ax = plt.subplots()
        ax.plot(vehicles_range, tot_pi0_vector, linewidth=2.0)
        ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
        ax.set_title(f"Total pi0 as function of vehicles number for {n_zones} zones")
        ax.set_xlabel("Number of vehicles")
        ax.set_ylabel("Total pi0")
        ax.grid()
        #plt.show()

def plot_bar_per_zone(n_zones, n_charging_stations, zones_with_charging_stations, city_grid_data, city_grid_CS_data, zones_id_dict):
    output_dir=f'img/{case}'
    zone_axis=[zones_id_dict[i] for i in np.arange(n_zones)]
    grid_with_CS=[zones_id_dict[i] for i in zones_with_charging_stations]
    cs_ticks=["z"+str(i) for i in grid_with_CS]
    #AVG VEHICLES
    fig11, (ax11, ax21) = plt.subplots(1,2)
    fig11.set_size_inches(18.5, 10.5)
    ax11.bar(zone_axis,city_grid_data['av_vehicles']) 
    #ax21.bar(np.arange(n_charging_stations),n_servers[n_zones:n_zones+n_charging_stations],label='number of servers', color='red', alpha=0.7)
    ax21.bar(np.arange(n_charging_stations),city_grid_CS_data['av_vehicles'], label='number of vehicles', alpha=0.7)
    fig11.suptitle(f"Average vehicles per zones")
    ax11.set_xlabel("Zones")
    ax21.set_xlabel("Charging Stations")
    ax11.set_ylabel("Vehicles")
    ax21.set_xticks(list(np.arange(n_charging_stations)))
    ax21.set_xticklabels(cs_ticks, rotation=45)
    ax11.grid()
    ax21.grid()
    ax21.legend()
    plt.savefig(f'{output_dir}/bar_veh.png')
    #plt.show()
    #IDLE+WAITING TIME
    fig12, (ax12,ax22, ax32) = plt.subplots(1,3)
    fig12.set_size_inches(18.5, 10.5)
    ax12.bar(zone_axis,city_grid_data['av_delay'])
    ax22.bar(np.arange(n_charging_stations),city_grid_CS_data['av_delay'])
    ax32.bar(np.arange(n_charging_stations),city_grid_CS_data['av_waiting'])
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
    ax22.set_xticks(list(np.arange(n_charging_stations)))
    ax22.set_xticklabels(cs_ticks)
    ax32.set_xticks(list(np.arange(n_charging_stations)))
    ax32.set_xticklabels(cs_ticks, rotation=45)
    ax12.grid()
    ax22.grid()
    ax32.grid()
    plt.savefig(f'{output_dir}/bar_del.png')
    #plt.show()
    #THROUGHPUT
    fig13, (ax13, ax23) = plt.subplots(1,2)
    fig13.set_size_inches(18.5, 10.5)
    ax13.bar(zone_axis,city_grid_data['throughput'])
    ax23.bar(np.arange(n_charging_stations), city_grid_CS_data['throughput'],color='red')
    fig13.suptitle(f"Vehicles througput per zones")
    ax13.set_xlabel("Zones")
    ax23.set_xlabel("Charging Stations")
    ax13.set_ylabel("Throughput")
    ax23.set_ylabel("Throughput")
    ax23.set_xticks(list(np.arange(n_charging_stations)))
    ax23.set_xticklabels(cs_ticks, rotation=45)
    ax13.grid()
    ax23.grid()
    plt.savefig(f'{output_dir}/bar_thr.png')
    #plt.show()
    #UTILIZATION
    fig14, (ax14, ax24) = plt.subplots(1,2, sharey=True)
    fig14.set_size_inches(18.5, 10.5)
    ax14.bar(zone_axis,city_grid_data['utilization']*100)
    ax24.bar(np.arange(n_charging_stations), city_grid_CS_data['utilization']*100, color='green')
    fig14.suptitle(f"Utilization per zones")
    ax14.set_xlabel("Zones")
    ax24.set_xlabel("Charging station")
    ax14.set_ylabel("Utilization [%]")
    ax24.set_xticks(list(np.arange(n_charging_stations)))
    ax24.set_xticklabels(cs_ticks, rotation=45)
    ax14.grid()
    ax24.grid()
    plt.savefig(f'{output_dir}/bar_ut.png')
    #plt.show()
    #UN DEMAND
    fig15, ax15 = plt.subplots()
    fig15.set_size_inches(18.5, 10.5)
    ax15.bar(zone_axis,city_grid_data['un_demand']*100)
    ax15.set_title(f"Unsatisfied mobility demand per zones")
    ax15.set_xlabel("Zones")
    ax15.set_ylabel("Unsatisfied demand [%]")
    ax15.grid()
    plt.savefig(f'{output_dir}/bar_und.png')
    #plt.show()
    #LOST REQ
    fig16, ax16 = plt.subplots()
    fig16.set_size_inches(18.5, 10.5)
    ax16.bar(zone_axis,city_grid_data['lost_req'])
    ax16.set_title(f"Lost mobility requests per zones")
    ax16.set_xlabel("Zones")
    ax16.set_ylabel("Lost requests")
    ax16.grid()
    plt.savefig(f'{output_dir}/bar_lost.png')
    #plt.show()

#COMPUTE FLUXES IN NETWORK WITH CS (k as function of total throughput or total distance)
def total_fluxes(n_zones, n_charging_stations, OD_matrix, zones_with_charging_stations, trips_autonomy):
    aug_matrix=np.zeros((n_zones+n_charging_stations,n_zones+n_charging_stations))
    aug_matrix[0:n_zones,0:n_zones]=OD_matrix
    #define only element in the CS row equal to one corresponding to its zone
    for j in range(n_charging_stations):
        aug_matrix[n_zones+j][zones_with_charging_stations[j]]=1
        #aug_matrix[0:n_zones,n_zones+j]=1
    #print("\nAUG:\n", aug_matrix)
    lambda_vec=np.random.rand(n_charging_stations+n_zones)
    lambda_vec=lambda_vec/(np.sum(lambda_vec))
    #print("lam: ",lambda_vec)
    #print("sumlam: ",np.sum(lambda_vec))
    #lambda_vec=np.ones((n_zones+n_charging_stations))/(n_zones+n_charging_stations) #initialize vector of flows
    num_it=0
    k=1/trips_autonomy
    #ITERATE TO FIND FLOWS AT STEADY STATE
    iterate=True
    while iterate:
        flows=np.dot(lambda_vec,aug_matrix)#compute vector of flows
        #print(f"flows it{num_it}: {flows}")
        tot_CS_flows=0
        for i in zones_with_charging_stations:
            tot_CS_flows+=flows[i]
        #print("CS flows: ",tot_CS_flows)
        tot_flows=np.sum(flows)
        #print("tot flows: ", tot_flows)
        k=(1/trips_autonomy)/(tot_CS_flows/tot_flows)
        #print("K: ", k)
        if n_charging_stations>0:
            #COMPUTE CHARGING FLOWS FOR EACH ZONE WITH CURRENT NETWORK FLOWS 
            for i in range(len(zones_with_charging_stations)):
                #correct OD matrix with k
                aug_matrix[0:n_zones,zones_with_charging_stations[i]]=OD_matrix[0:n_zones,zones_with_charging_stations[i]]*(1-k)
                aug_matrix[0:n_zones,n_zones+i]=OD_matrix[0:n_zones,zones_with_charging_stations[i]]*k
            #print("Aug:\n", aug_matrix)
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
                    #print("AAAAAAAAAA")
                    lambda_vec=flows
                    num_it+=1
                else:
                    iterate=False
                    flows=np.array([max(0,i) for i in flows])
            else:
                iterate=False
                flows=np.array([])
    #print("augmented OD matrix:\n", aug_matrix)
    #print("num it: ", num_it)
    
    return flows, aug_matrix
"""
    #old version
    ## RELATIVE FLOWS COMPUTATION for each zone and chargin station(approximation)
    flows, aug_OD_matrix=fluxes(n_zones,n_charging_stations,OD_matrix,zones_with_charging_stations, dist_matrix, service_rates, dist_autonomy)
    print(f"\nRelative flows per zone (last {n_charging_stations} refer(s) to charging station(s))", flows)
    print("Complete OD matrix (with CS):\n", np.round(aug_OD_matrix,4))
    ## APPROXIMATE CALCULATION OF CHARGING RATES 
    charging_flows=compute_charging_flows(n_zones,flows,aug_OD_matrix,dist_matrix,dist_autonomy) #charging rates at steady state
    #print("charging rates: ",charging_flows)
    """
def fluxes_with_reloc4charg(n_zones, n_charging_stations, OD_matrix, zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, closest_CS=False):
    aug_matrix=np.zeros((n_zones+n_charging_stations,n_zones+n_charging_stations))
    aug_matrix[0:n_zones,0:n_zones]=OD_matrix
    zones_list=list(np.arange(0,n_zones))
    zones_without_CS=[zone for zone in zones_list if zone not in zones_with_charging_stations]
    #define only element in the CS row equal to one corresponding to its zone
    for j in range(n_charging_stations):
        aug_matrix[n_zones+j][zones_with_charging_stations[j]]=1
        #aug_matrix[0:n_zones,n_zones+j]=1
    #print("\nAUG:\n", aug_matrix)
    lambda_vec=np.random.rand(n_charging_stations+n_zones)
    lambda_vec=lambda_vec/(np.sum(lambda_vec))
    #print("lam: ",lambda_vec)
    #print("sumlam: ",np.sum(lambda_vec))
    #lambda_vec=np.ones((n_zones+n_charging_stations))/(n_zones+n_charging_stations) #initialize vector of flows
    num_it=0
    k=1/trips_autonomy
    if n_charging_stations>0:
        #redirect flow from each zone to CS
        if closest_CS:
            closest_CS_ids=find_closest_CS(zones_with_charging_stations, city_grid, zones_id_dict)
            inv_zones_id_dict = {v: k for k, v in zones_id_dict.items()}
            closest_CS_od=[inv_zones_id_dict[i] for i in closest_CS_ids]
            zones_with_charging_stations=np.array(zones_with_charging_stations)
            for i in range(n_zones):
                aug_matrix[0:n_zones,n_zones+(np.where(zones_with_charging_stations==closest_CS_od[i])[0].item())]+=OD_matrix[0:n_zones,i]*k
                #aug_matrix[0:n_zones,i]=OD_matrix[0:n_zones,i]*(1-k)
        else:
            for i in range(len(zones_with_charging_stations)):
                #correct OD matrix with k
                #fluxes directed to the zone with CS inside
                aug_matrix[0:n_zones,n_zones+i]=OD_matrix[0:n_zones,zones_with_charging_stations[i]]*k
                for j in range(len(zones_without_CS)):
                    #fluxes directed to zones WITHOUT CS inside
                    aug_matrix[0:n_zones,n_zones+i]=aug_matrix[0:n_zones,n_zones+i]+OD_matrix[0:n_zones,zones_without_CS[j]]*(k/n_charging_stations)
        #for both each mobility zone times (1-k)
        aug_matrix[0:n_zones,0:n_zones]=OD_matrix[0:n_zones,0:n_zones]*(1-k)        
        #print("Aug:\n", aug_matrix)
        #print(np.sum(aug_matrix[0,:]))
    #ITERATE TO FIND FLOWS AT STEADY STATE
    iterate=True
    while iterate:
        flows=np.dot(lambda_vec,aug_matrix)#compute vector of flows
        #print(f"flows it{num_it}: {flows}")
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
                    #print("AAAAAAAAAA")
                    lambda_vec=flows
                    num_it+=1
                else:
                    iterate=False
                    flows=np.array([max(0,i) for i in flows])
            else:
                iterate=False
                flows=np.array([])
    #print("augmented OD matrix:\n", aug_matrix)
    #print("num it: ", num_it)
    return flows, aug_matrix

def find_closest_CS(zones_with_CS, city_grid, zones_id_dict):
    grid_with_CS=[zones_id_dict[i] for i in zones_with_CS]
    grid_with_CS_geo=city_grid.loc[city_grid.index.isin(grid_with_CS)]['geometry']
    grid_with_CS_geo=grid_with_CS_geo.to_crs(epsg=3035)
    city_grid=city_grid.to_crs(epsg=3035)
    #grid_with_CS_geo=grid_with_CS_geo.centroid
    nearest_CS=[]
    for zone in city_grid.geometry:
        nearest_CS.append(grid_with_CS_geo.distance(zone.centroid).sort_values().index[0])
    return nearest_CS
#COPUTE FLUXES IN NETWORK WITH CS (old version)
def fluxes(n_zones, n_charging_stations, OD_matrix, zones_with_charging_stations, dist_matrix, service_rates, dist_autonomy):
    #compute fluxes including approximation for charging-station queues
    #define an augmented OD matrix to contain the charging zones
    aug_matrix=np.zeros((n_zones+n_charging_stations,n_zones+n_charging_stations))
    aug_matrix[0:n_zones,0:n_zones]=OD_matrix
    #define only element in the CS row equal to one corresponding to its zone
    for j in range(n_charging_stations):
        aug_matrix[n_zones+j][zones_with_charging_stations[j]]=1
    #print("\nAUG:\n", aug_matrix)
    lambda_vec=np.ones((n_zones+n_charging_stations)) #initialize vector of flows
    num_it=0
    #ITERATE TO FIND FLOWS AT STEADY STATE
    iterate=True
    while iterate:
        flows=np.dot(lambda_vec,aug_matrix) #compute vector of flows
        if n_charging_stations>0:
            #COMPUTE CHARGING FLOWS FOR EACH ZONE WITH CURRENT NETWORK FLOWS 
            charging_flows=compute_charging_flows(n_zones, flows, aug_matrix, dist_matrix, dist_autonomy)
            for i in range(n_charging_stations):
                """
                #alpha given by charging rate in zone i/service rate in zone i
                #alpha=charging_flows[zones_with_charging_stations[i]]/service_rates[zones_with_charging_stations[i]]
                #print("alpha: ",alpha)
                #avg_flow=np.sum(flows)/(n_zones+n_charging_stations)
                #correction_coefficient=alpha*avg_flow/(n_zones+n_charging_stations)
                #aug_matrix[0:n_zones,n_zones+i]=flows[0:n_zones]*correction_coefficient
                """
                #DEFINE ENTRIES OF OD MATRIX CORRESPONDING TO CS COLUMNS AS FUNCTION OF COMPUTE CHARGING FLOWS
                #avg_network_flow=np.sum(flows)/(n_zones+n_charging_stations)
                aug_matrix[0:n_zones,n_zones+i]=charging_flows[zones_with_charging_stations[i]]/(flows[0:n_zones]*n_zones)
            #IMPOSE STHOCASTICITY
            aug_matrix=aug_matrix/aug_matrix.sum(axis=1)[:,None]
        #print("AUUUG:\n",aug_matrix)
        #print("flooows:\n", flows)
        if (abs(flows-lambda_vec)>10e-4).any(): #check on convergence
            lambda_vec=flows
            num_it+=1
        else:
            iterate=False
    #print("charging rates: ",charging_flows)
    #print("augmented OD matrix:\n", aug_matrix)
    print("num it: ", num_it)
    return flows, aug_matrix

def product_form_distribution(n_zones, n_charging_stations, n_vehicles, flows, service_rates, n_servers):
    ##PRODUCT FORM JOINT DISTRIBUTION CALCULATION
        print("\n\ndistribution results:")
        #convolutional algorithm for the normalization constant (used for distribution calculations)
        relative_utilization=np.divide(flows,service_rates)
        print("Relative rho: ", relative_utilization)
        normalization_constant=compute_normalization_constant_MS(n_zones+n_charging_stations, n_vehicles, service_rates, relative_utilization, n_servers)
        print("Normalization constant: ", normalization_constant)

        #Distribution calculation methods (recursive and combinatorial)
        t1=time.time()
        counters = [0] * (n_zones+n_charging_stations)
        state_list = list()
        find_states(counters, n_vehicles, n_zones+n_charging_stations, 0, state_list)
        #print("States: ", state_list)
        #empty_queues, tot_pi0, tot_requests_lost=compute_pi0_rec(state_list, relative_utilization, normalization_constant, service_rates)
        print("t rec: ", time.time()-t1)

        ## Compute metrics from distribution
        avg_vehicles=np.zeros(n_zones+n_charging_stations)
        avg_waiting=np.zeros(n_zones+n_charging_stations)
        states_pi=[]
        for state in state_list:
            state=np.array(state)
            pi_state=compute_generic_pi(n_zones+n_charging_stations, state, relative_utilization, normalization_constant, n_servers)
            states_pi.append((state,pi_state))
            avg_vehicles+=state*pi_state        
        #print("STATES:\n",states_pi)
        
        print("\nAvg vehicles per zone: ", avg_vehicles)

def plot_city_zones(grid, zones_with_charging_stations_ID, annotate=False):
        fig, ax = plt.subplots()
        plt.title("")
        plt.xlabel(None)
        plt.xticks([])
        plt.ylabel(None)
        plt.yticks([])
        grid.plot(color="white", edgecolor="black", ax=ax)
        #CS_zones=grid.loc[grid['zone_id'].isin(zones_with_charging_stations_ID)]
        #CS_zones.plot(color="red", edgecolor="black", ax=ax, label='zones with Charging Stations')
        grid['coords'] = grid['geometry'].apply(lambda x: x.centroid.coords[0])
        if annotate:
            for idx, row in grid.iterrows():
                plt.annotate(
                    text=row['zone_id'], xy=row['coords'], horizontalalignment='center'
                )
        plt.legend()
        #plt.show()

def generate_data(n_zones):
    n_zones=n_zones
    zone_rates=np.random.randint(low=1, high=5, size=n_zones)
    OD_matrix=build_OD(n_zones, True)
    city_grid=None

    return city_grid, n_zones, OD_matrix, zone_rates

def create_grid_and_map_data(city_name, cities_info_file, data_csv_file,zone_size):
    cities=pd.read_csv(cities_info_file)
    city_info=cities.loc[(cities.city==city_name)]
    print(city_info)
    minLon= city_info.minLon.values[0]
    minLat= city_info.minLat.values[0]
    shiftLon= city_info.ShiftLon500m.values[0]
    shiftLat= city_info.ShiftLat500m.values[0]
    maxLon=city_info.maxLon.values[0]
    maxLat=city_info.maxLat.values[0]
    bookings_df=pd.read_csv(data_csv_file)
    #bookings_df=bookings_df.loc[(bookings_df['duration']>3*60) & (bookings_df['duration']<60*60)]
    #bookings_df=bookings_df.loc[bookings_df['distance']>500]
    bookings_df=bookings_df.loc[(bookings_df['init_lon']>minLon)&(bookings_df['init_lon']<maxLon)]
    bookings_df=bookings_df.loc[(bookings_df['final_lon']>minLon)&(bookings_df['final_lon']<maxLon)]
    bookings_df=bookings_df.loc[(bookings_df['init_lat']>minLat)&(bookings_df['init_lat']<maxLat)]
    bookings_df=bookings_df.loc[(bookings_df['final_lat']>minLat)&(bookings_df['final_lat']<maxLat)]
    rows, cols, grid = get_city_grid_as_gdf(
            (
                min(bookings_df.init_lon.min(), bookings_df.final_lon.min()),
                min(bookings_df.init_lat.min(), bookings_df.final_lat.min()),
                max(bookings_df.init_lon.max(), bookings_df.final_lon.max()),
                max(bookings_df.init_lat.max(), bookings_df.final_lat.max())
            ),"epsg:4326", zone_size
        )
    grid["zone_id"] = grid.index.values
    grid.to_pickle("new_grid.pickle")
    #associate zone id to bookings origin and destination
    origin_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(bookings_df['init_lon'], bookings_df['init_lat'], crs="EPSG:4326"))
    trips_origin=gpd.sjoin(origin_points,grid,how='left',op='intersects')
    bookings_df["origin_id"] = trips_origin['zone_id']
    bookings_df['origin_point']= trips_origin.geometry
    dest_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(bookings_df['final_lon'], bookings_df['final_lat'], crs="EPSG:4326"))
    trips_dest=gpd.sjoin(dest_points,grid,how='left',op='intersects')
    bookings_df["destination_id"] = trips_dest.zone_id
    bookings_df['dest_point']= trips_dest.geometry
    bookings_df=bookings_df.dropna()
    bookings_df['line'] = bookings_df.apply(lambda row: LineString([row['origin_point'], row['dest_point']]), axis=1)
    bookings_df.to_csv("trips_with_zone_id.csv")
    return grid
    
def get_city_grid_as_gdf(total_bounds, crs, bin_side_length):
    x_min, y_min, x_max, y_max = total_bounds

    # this has to be pretty much the same long all the longitudes, cause paralles are equidistant, so equal for every city
    p1 = (y_min, x_min)
    p2 = (y_min + 0.01, x_min)
    height_001 = haversine(p1, p2, unit=Unit.METERS)
    height = (0.01 * bin_side_length) / height_001

    # width changes depending on the city, casuse longitude distances vary depending on the latitude.
    p1 = (y_min, x_min)
    p2 = (y_min, x_min + 0.01)
    width_001 = haversine(p1, p2, unit=Unit.METERS)
    width = (0.01 * bin_side_length) / width_001

    rows = int(np.ceil((y_max - y_min) / height))
    cols = int(np.ceil((x_max - x_min) / width))
    x_left = x_min
    x_right = x_min + width
    polygons = []
    for i in range(cols):
        y_top = y_max
        y_bottom = y_max - height
        for j in range(rows):
            polygons.append(Polygon([(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)]))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left = x_left + width
        x_right = x_right + width
    grid = gpd.GeoDataFrame({"geometry": polygons})
    grid["zone_id"] = range(len(grid))
    grid.crs = crs
    return rows, cols, grid
    
def get_data_from_file(grid_pickle_file, data_pickle_file):
    bookings_data=pd.read_pickle(data_pickle_file)
    if grid_pickle_file!=None:
        city_grid = pd.read_pickle(grid_pickle_file)
        #print(city_grid_p)
        zones_id=city_grid['zone_id'].values.tolist()
        zones_id.sort()
        n_zones=len(zones_id)
    else:
        city_grid=None
        zones_id=bookings_data['origin_id'].unique()
        zones_id.sort()
        n_zones=len(zones_id)
    zones_id_dict={} #map index of OD matrix to zone indexes
    for i in range(n_zones):
        zones_id_dict[i]=zones_id[i]
    #print(zones_id_dict)
    inv_zones_id_dict = {v: k for k, v in zones_id_dict.items()}
    
    #filter data
    bookings_data=bookings_data.loc[bookings_data['daytype']=='weekday']
    #hour_count=bookings_data.groupby(['start_hour']).size()
    #print(hour_count)
    bookings_data=bookings_data.loc[bookings_data['start_hour'].isin([14,15,16])]
    bookings_data=bookings_data.loc[(bookings_data['duration']>3*60) & (bookings_data['duration']<60*60)]
    bookings_data=bookings_data.loc[bookings_data['euclidean_distance']>500]
    bookings_data=bookings_data.loc[bookings_data['avg_speed_kmh']<120]
    #df for OD
    bookings_data_OD=bookings_data[['origin_id','destination_id']]
    bookings_data_OD_grouped=bookings_data_OD.groupby(['origin_id','destination_id']).size().reset_index(name='counts')
    #a=bookings_data_OD.origin_id.value_counts()
    #bookings_data_OD_grouped['rel_count']=bookings_data_OD_grouped['counts'].div(bookings_data_OD_grouped['origin_id'].map(a))
    #create OD
    OD_matrix=np.zeros((n_zones,n_zones))
    for index, row in bookings_data_OD_grouped.iterrows():
        OD_matrix[inv_zones_id_dict[row['origin_id']],inv_zones_id_dict[row['destination_id']]]=row['counts']
    zones_service_rates=np.zeros(n_zones)
    for i in range (n_zones):
        if np.sum(OD_matrix[i,:])==0:
            OD_matrix[i][i]=1
    for i in range(n_zones):
        zones_service_rates[i]=np.sum(OD_matrix[i,:])

    OD_matrix=OD_matrix/OD_matrix.sum(axis=1)[:,None]
     
    return city_grid, n_zones, zones_id_dict, OD_matrix, zones_service_rates

def get_balance_OD(grid_pickle, data_csv):
    bookings_df=pd.read_csv(data_csv)
    #get zones
    zones_id=np.unique(bookings_df[['origin_id', 'destination_id']].values).astype(int)
    zones_id=np.sort(zones_id)
    n_zones=zones_id.size
    #get grid with only valid zones
    if grid_pickle!=None:
        city_grid = pd.read_pickle(grid_pickle)
        city_grid=city_grid.loc[city_grid['zone_id'].isin(zones_id)]
    else:
        city_grid=None
    #create dicr of zones for OD-grid
    zones_id_dict={} #map index of OD matrix to zone indexes
    for i in range(n_zones):
        zones_id_dict[i]=zones_id[i]
    inv_zones_id_dict = {v: k for k, v in zones_id_dict.items()}
    #filter bookings data
    bookings_df=bookings_df.loc[(bookings_df['duration']>3*60) & (bookings_df['duration']<60*60)]
    bookings_df=bookings_df.loc[bookings_df['distance']>500]
    #df for OD
    bookings_data_OD=bookings_df[['origin_id','destination_id']]
    bookings_data_OD_grouped=bookings_data_OD.groupby(['origin_id','destination_id']).size().reset_index(name='counts')
    OD_matrix=np.zeros((n_zones,n_zones))
    for index, row in bookings_data_OD_grouped.iterrows():
        OD_matrix[inv_zones_id_dict[row['origin_id']],inv_zones_id_dict[row['destination_id']]]=row['counts']
    OD_matrix=OD_matrix/OD_matrix.sum(axis=1)[:,None]

    #df for demand rates
    bookings_df['start_date'] = pd.to_datetime(bookings_df['init_time'],unit='s').dt.date
    bookings_df['start_hour'] = pd.to_datetime(bookings_df['init_time'],unit='s').dt.hour
    bookings_df_rate=bookings_df.groupby(['origin_id','start_date','start_hour']).size().reset_index(name='counts')
    bookings_df_rate=bookings_df_rate.groupby(['origin_id','start_date']).agg(date_mean=("counts",'mean'))
    bookings_df_rate=bookings_df_rate.groupby('origin_id').agg(mean_service_time=('date_mean','mean'))
    #print(bookings_df_rate)
    zones_service_rates=np.array(bookings_df_rate['mean_service_time'])
    
     
    return city_grid, n_zones, zones_id_dict, OD_matrix, zones_service_rates

def get_hour_OD(grid_pickle, data_csv, hour_of_day, week_day=False):
    bookings_df=pd.read_csv(data_csv)
    print("\nNumber of entries in original dataframe: ", len(bookings_df.index))

    #filter bookings data
    bookings_df=bookings_df.loc[(bookings_df['duration']>3*60) & (bookings_df['duration']<60*60)]
    bookings_df=bookings_df.loc[bookings_df['distance']>500]
    print("Number of entries in dataframe (real trips): ", len(bookings_df.index))
    bookings_df['start_date'] = pd.to_datetime(bookings_df['init_time'],unit='s').dt.date
    bookings_df['start_hour'] = pd.to_datetime(bookings_df['init_time'],unit='s').dt.hour
    if week_day:
        bookings_df=bookings_df.loc[bookings_df['weekday'].isin([0,1,2,3,4])]
    try:
        bookings_df=bookings_df.loc[bookings_df['start_hour']==hour_of_day]
        interval_size=1
        print(f"Single hour bookings h:{hour_of_day}")
    except:
        bookings_df=bookings_df.loc[bookings_df['start_hour'].isin(hour_of_day)]
        interval_size=len(hour_of_day)
        print(f"Interval of hours bookings h:{hour_of_day}")
    print("Number of entries in filtered dataframe: ", len(bookings_df.index))

    #get zones
    zones_id=np.unique(bookings_df[['origin_id', 'destination_id']].values).astype(int)
    zones_id=np.sort(zones_id)
    n_zones=zones_id.size
    #get grid with only valid zones
    if grid_pickle!=None:
        city_grid = pd.read_pickle(grid_pickle)
        city_grid=city_grid.loc[city_grid['zone_id'].isin(zones_id)]
    else:
        city_grid=None
    #create dicr of zones for OD-grid
    zones_id_dict={} #map index of OD matrix to zone indexes
    for i in range(n_zones):
        zones_id_dict[i]=zones_id[i]
    inv_zones_id_dict = {v: k for k, v in zones_id_dict.items()}
    
    #df for OD
    bookings_data_OD=bookings_df[['origin_id','destination_id']]
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

    OD_matrix=OD_matrix/OD_matrix.sum(axis=1)[:,None]

    #df for service rates
    bookings_df_rate=bookings_df.groupby(['origin_id','start_date','start_hour']).size().reset_index(name='counts')
    #print(bookings_df_rate)
    bookings_df_rate=bookings_df_rate.groupby(['origin_id','start_date']).agg(date_mean=("counts",'sum'))
    bookings_df_rate['date_mean']=bookings_df_rate['date_mean']/interval_size
    #print(bookings_df_rate)
    bookings_df_rate=bookings_df_rate.groupby('origin_id').agg(mean_service_time=('date_mean','mean'))
    #print(bookings_df_rate)
    bookings_df_rate=bookings_df_rate.reindex(zones_id, fill_value=0.1)
    #zones_service_rates=np.zeros(n_zones)
    zones_service_rates=np.array(bookings_df_rate['mean_service_time'])
   
    return city_grid, n_zones, zones_id_dict, OD_matrix, zones_service_rates, tot_departure_per_zone, tot_arrival_per_zone

def get_data_from_OD_file(csv_file):
    city_grid=None
    OD=pd.read_csv(csv_file, sep=";")
    OD = OD.iloc[: , 1:]
    zones_id=OD.columns.values.tolist()
    n_zones=len(zones_id)
    print(n_zones)
    OD_matrix=OD.to_numpy()
    print(OD_matrix.shape)
    zone_rates=np.zeros(n_zones)
    
    for i in range(n_zones):
        zone_rates[i]=np.sum(OD_matrix[i,:]) 
        if zone_rates[i]==0:
            zone_rates[i]=0.1
    #print(zone_rates)
    for i in range(n_zones):
        if np.sum(OD_matrix[i,:])!=0:
            OD_matrix[i]=OD_matrix[i]/np.sum(OD_matrix[i])
        else:
            OD_matrix[i,i]=1
    return city_grid, n_zones, OD_matrix, zone_rates

def print_data(n_zones, n_vehicles, trips_autonomy, dist_autonomy, outlet_rate, zones_with_charging_stations, outlet_per_stations):
        print("Number of zones: ", n_zones)
        print("Number of vehicles: ", n_vehicles)
        print("Trips autonomy: ", np.round(trips_autonomy,2))
        print(f"Distance autonomy: {np.round(dist_autonomy,2)} km")
        print(f"Charging rates: {np.round(outlet_rate,2)}/h")
        print("Zones with charging stations", zones_with_charging_stations)
        print("Charging outlet per station: ", outlet_per_stations)

def steady_state_analysis(n_zones, n_charging_stations,  n_vehicles, service_rates, n_servers, OD_matrix, zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, reloc4charg=False, closest_CS=False):
    if reloc4charg:
        flows, aug_OD_matrix=fluxes_with_reloc4charg(n_zones, n_charging_stations, OD_matrix, zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, closest_CS)
    else:
        flows, aug_OD_matrix=total_fluxes(n_zones, n_charging_stations, OD_matrix, zones_with_charging_stations, trips_autonomy)
    if flows.size!=0:
        ## MS_MVA - Multi-Server Mean Value Analysis of the network
        av_vehicles, av_delay, ov_throughput=MS_MVA(n_zones+n_charging_stations, n_vehicles, service_rates, flows, n_servers)
        throughput_vector=ov_throughput*flows
        av_waiting=av_delay-(1/service_rates)
        #compute utilization vector (rho) with computed flows and service rates
        rho=np.divide(throughput_vector,np.multiply(service_rates,n_servers))
        #Compute unsatisfied demand and mobility requests for mobility zones only
        unsatisfied_demand_per_zone=((1-rho[0:n_zones]))
        lost_requests_per_zone=np.round(np.multiply(unsatisfied_demand_per_zone,service_rates[0:n_zones]),4)

        zones_with_charging_stations_ID=[]
        for zone in zones_with_charging_stations:
            zones_with_charging_stations_ID.append(zones_id_dict[zone])  
        #build df with statistics over city grid
        city_grid_data=city_grid.copy()
        city_grid_CS_data=city_grid.loc[city_grid['zone_id'].isin(zones_with_charging_stations_ID)]
        city_grid_data=city_grid_data.sort_values(by=['zone_id'])
        city_grid_CS_data=city_grid_CS_data.sort_values(by=['zone_id'])
        city_grid_data['service_rates']=service_rates[0:n_zones]
        city_grid_CS_data['service_rates']=service_rates[n_zones:n_zones+n_charging_stations]
        city_grid_data['flows']=flows[0:n_zones]
        city_grid_CS_data['flows']=flows[n_zones:n_zones+n_charging_stations]
        city_grid_data['throughput']=throughput_vector[0:n_zones]
        city_grid_CS_data['throughput']=throughput_vector[n_zones:n_zones+n_charging_stations]
        city_grid_data['av_vehicles']=av_vehicles[0:n_zones]
        city_grid_CS_data['av_vehicles']=av_vehicles[n_zones:n_zones+n_charging_stations]
        city_grid_data['av_delay']=av_delay[0:n_zones]
        city_grid_CS_data['av_delay']=av_delay[n_zones:n_zones+n_charging_stations]
        city_grid_CS_data['av_waiting']=av_waiting[n_zones:n_zones+n_charging_stations]
        city_grid_data['utilization']=rho[0:n_zones]
        city_grid_CS_data['utilization']=rho[n_zones:n_zones+n_charging_stations]
        city_grid_data['un_demand']=unsatisfied_demand_per_zone[0:n_zones]*100
        city_grid_data['lost_req']=lost_requests_per_zone[0:n_zones]
    else:
        city_grid_data=pd.DataFrame()
        city_grid_CS_data=pd.DataFrame()
        ov_throughput=None

    return city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput

def compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, city_grid_CS_data):
    rho=np.array(city_grid_CS_data['utilization'])
    throughput_vector=np.array(city_grid_CS_data['throughput'])
    #PROBABILITY TO WAIT IN CS (Erlang-C)
    waiting_p=np.zeros(n_charging_stations)
    for i in range(n_charging_stations):
        num=1/(np.math.factorial(n_servers[n_zones+i]))*(throughput_vector[i]/service_rates[n_zones+i])**n_servers[n_zones+i]*(1/(1-rho[i]))
        den=0
        for m in range(int(n_servers[n_zones+i])):
            den+=1/(np.math.factorial(m))*(throughput_vector[i]/service_rates[n_zones+i])**m
        den+=1/(np.math.factorial(n_servers[n_zones+i]))*(throughput_vector[i]/service_rates[n_zones+i])**n_servers[n_zones+i]*(1/(1-rho[i]))
        waiting_p[i]=num/den
    return waiting_p

def plot_data_on_grid(city_grid_data, city_grid_CS_data, ov_throughput):
    #fig1, (ax11,ax12,ax13)=plt.subplots(1,3)
    #city_grid_data.plot(color="white", edgecolor="black")
    av_ud=np.sum(np.multiply(np.array(city_grid_data['un_demand']),(np.array(city_grid_data['service_rates']/np.sum(np.array(city_grid_data['service_rates']))))))
    #plot over city grid
    fig7, (ax11,ax12,ax13)=plt.subplots(1,3)
    fig7.set_size_inches(18.5, 10.5)
    fig7.text(0.5, 0.98, f"System throughput: {np.round(ov_throughput,2)}\nAverage unsatisfied demand: {np.round(av_ud,2)}%",horizontalalignment='center', verticalalignment='top')
    city_grid_CS_data=city_grid_CS_data.to_crs(epsg=3035)
    CS_centroids=city_grid_CS_data.centroid
    CS_centroids=CS_centroids.to_crs(epsg=4326)
    city_grid_data.plot(column="av_vehicles", ax=ax11, legend=True, legend_kwds={'label':'Average vehicles'},edgecolor="black")
    CS_centroids.plot(ax=ax11, marker='.', color="red", markersize=24)
    city_grid_data.plot(column="un_demand", ax=ax12, legend=True, legend_kwds={'label':'Unsatisfied demand [%]'},edgecolor="black")
    CS_centroids.plot(ax=ax12, marker='.', color="red", markersize=24)
    city_grid_data.plot(column="throughput", ax=ax13, legend=True, legend_kwds={'label':'Throughput'},edgecolor="black")
    CS_centroids.plot(ax=ax13, marker='.', color="red", markersize=24)
    ax11.set_axis_off()
    ax12.set_axis_off()
    ax13.set_axis_off()
    output_dir=f'img/{case}'
    mkdir_p(output_dir)
    plt.savefig(f'{output_dir}/map_plot.png')
    print("ov throughput: ", ov_throughput)
    print("avg unsatisfied demand: ", av_ud)
    #plt.show()
    return 

def print_stat(city_grid_data, city_grid_CS_data, ov_throughput):
        print("Relative flows per zone:\n", city_grid_data['flows'])
        print("Relative flows in CS:\n", city_grid_CS_data['flows'])
        print("\nAverage # vehicles per zone:\n", np.round(city_grid_data['av_vehicles'],3))
        print("\nAverage # vehicles in CS:\n", np.round(city_grid_CS_data['av_vehicles'],3))
        #print("Average vehicles idle time vector: ", np.round(av_delay,3))
        #print('Average vehicles "waiting" time vector: ', np.round(av_waiting,3))
        print("Overall throughput (constant for the flow calculation): ", np.round(ov_throughput,4))
        print("Throughputs per zone:\n", np.round(city_grid_data['throughput'],3))
        print("Throughputs in CS:\n", np.round(city_grid_CS_data['throughput'],3))
        print("Utilization per zone:\n", np.round(city_grid_data['utilization'],3))
        print("Utilization per CS:\n", np.round(city_grid_CS_data['utilization'],3))
        print(f"Waiting probability in CS:\n {np.round(city_grid_CS_data['waiting_p']*100,3)}%")

def plot_results(city_grid_data, city_grid_CS_data, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, zones_id_dict, zones_with_charging_stations):
        output_dir=f'img/{case}'
        flows=np.array(city_grid_data['flows'])
        _aug_OD=aug_OD_matrix
        fig4, ax4 = plt.subplots()
        fig4.set_size_inches(18.5, 10.5)
        for _n_vehicles in range_vehicles:
            _av_vehicles, _av_delay, _ov_throughput=MS_MVA(n_zones, _n_vehicles, service_rates[0:n_zones], flows, n_servers[0:n_zones])
            _throughput_vector=_ov_throughput*flows
            _rho=np.round(np.divide(_throughput_vector,np.multiply(service_rates[0:n_zones],n_servers[0:n_zones])),4)
            _unsatisfied_demand_per_zone=((1-_rho))
            ax4.hist(_unsatisfied_demand_per_zone, bins=30, cumulative=True, density=True, histtype='step', label=f'n vehicles: {_n_vehicles}')
        ax4.set_title(f"Cumulative distribution of unsatisfied mobility demand per zones with {n_zones} zones")
        ax4.set_xlabel("Unsatisfied demand")
        ax4.grid()
        ax4.legend(loc='upper left')
        plt.savefig(f'{output_dir}/cum_dist_range.png')
        #plt.show()
        grid_with_CS=[zones_id_dict[i] for i in zones_with_charging_stations]
        cs_ticks=["z"+str(i) for i in grid_with_CS]
        
        fig6, (ax61,ax62)=plt.subplots(2,1, sharex=True)
        fig6.set_size_inches(18.5, 10.5)
        ax61.bar(np.arange(n_charging_stations),city_grid_CS_data['av_vehicles'], alpha=.5)
        ax62.bar(np.arange(n_charging_stations),city_grid_CS_data['waiting_p']*100, color='green')
        fig6.suptitle("Vehicles and waiting proability in charging stations")
        ax61.set_title("Average number of vehicles per charging station")
        ax62.set_title("Vehicles waiting probability per charging station")
        ax62.set_xlabel("Charging stations")
        ax61.set_ylabel("Vehicles")
        ax62.set_ylabel("Probability [%]")
        ax61.set_xticks(list(np.arange(n_charging_stations)))
        ax61.set_xticklabels(cs_ticks, rotation=45)
        ax62.set_xticks(list(np.arange(n_charging_stations)))
        ax62.set_xticklabels(cs_ticks, rotation=45)
        ax61.grid()
        ax62.grid()
        plt.savefig(f'{output_dir}/wait_p.png')
        #ax61.scatter(np.arange(n_charging_stations),n_servers[n_zones:n_zones+n_charging_stations], label=f'n servers', color='red',marker='.')
        #fig6.legend()
        #plt.show()

        #cumulative distribution histogram of unsatisfied demand
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(18.5, 10.5)
        ax3.hist(city_grid_data['un_demand'], bins=30, cumulative=True, density=True, histtype='stepfilled' )
        ax3.set_title(f"Cumulative distribution of unsatisfied mobility demand per zones {n_vehicles} vehicles")
        ax3.set_xlabel("Unsatisfied demand")
        ax3.grid()
        plt.savefig(f'{output_dir}/cum_dist.png')
        #plt.show()

def place_CS_by(trips_data, zones_id_dict, n_charging_stations, order_by='arrival_per_zone', ascending_order=False):
        ordered_df_column=trips_data.sort_values(by=[order_by], ascending=ascending_order)[[order_by]]
        inv_zones_id_dict = {v: k for k, v in zones_id_dict.items()}
        #neighbors_delta=[1,-1,40,-40,39,41,-39,-41]
        neighbors_delta=[1,-1,40,-40,39,41,-39,-41,2,-2,41,38,-41,-38,-82,-81,-80,-79,-78,78,79,80,81,82]
        grid_with_charging_stations=[]
        flag=0
        for idx, row in ordered_df_column.iterrows():
            for i in grid_with_charging_stations:
                if idx in [i+j for j in neighbors_delta]:
                    flag=1
                    break
            if flag==0:
                grid_with_charging_stations.append(idx)
                if len(grid_with_charging_stations)==n_charging_stations:
                    break
            flag=0
        #print(grid_with_charging_stations)
        zones_with_charging_stations=[inv_zones_id_dict[i] for i in grid_with_charging_stations]
        return zones_with_charging_stations

def plot_gloal_indexes(ov_throughput, ov_ud):
    print(len(ov_ud))
    x_ax1=np.arange(0,len(ov_throughput))
    x_ax2=np.arange(0,len(ov_ud))
    fig1,(ax11, ax12)=plt.subplots(2,1, sharex=True)
    ax11.plot(x_ax1,ov_throughput)
    ax12.plot(x_ax2, ov_ud)
    fig1.suptitle("Global indexes for the system with hourly data")
    ax11.set_title("Overal throughput")
    ax12.set_title("Average unsatisfied demand")
    ax12.set_xlabel("Hour of the day for data")
    ax11.set_ylabel("System throughput")
    ax12.set_ylabel("Probability [%]")
    ax11.set_ylim([max(min(ov_throughput)-20,0), max(ov_throughput)+20])
    ax12.set_ylim([0,110])
    ax11.grid()
    ax12.grid()

if __name__=="__main__":
    np.random.seed(12)
    np.set_printoptions(suppress=True)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plot=False
    print_input_data=True
    print_results=False

    #city_grid=create_grid_and_map_data("Torino","info_per_city.csv","Turin_data_with_zones.csv",500)
    
    #GENERATE OR GET DATA FROM OD OR BOOKINGS AND GRID FILE
    #city_grid, n_zones, OD_from_file, zone_rates_from_file=generate_data(300)
    #city_grid, n_zones, OD_from_file, zone_rates_from_file=get_data_from_OD_file("L_12_OD.csv")
    #city_grid, n_zones, zones_id_dict, OD_from_file, zone_rates_from_file=get_data_from_file("city_grid.pickle","bookings_train.pickle")

    #city_grid, n_zones, zones_id_dict, OD_from_file, zone_rates_from_file=get_balance_OD("new_grid.pickle","trips_with_zone_id.csv")
    data_time_interval=np.arange(12,13)
    week_day=False
    city_grid, n_zones, zones_id_dict, OD_from_file, zone_rates_from_file, tot_departure_per_zone, tot_arrival_per_zone=get_hour_OD("new_grid.pickle","trips_with_zone_id.csv",data_time_interval,week_day)
    if week_day:
        case_time='wd_'
    else:
        case_time=''
    if data_time_interval.size==1:
        case_time+='h'+str(data_time_interval[0])
    else:
        if data_time_interval.size==24:
            case_time+='day'
        else:
            case_time+='h'+str(data_time_interval[0])+'-'+str(data_time_interval[data_time_interval.size-1])
    #plot_city_zones(city_grid,[],True)
    
    #n_zones=300
    n_vehicles=400
    range_vehicles=[300,500,700]
    n_charging_stations=10
    #dist_autonomy=300
    #range_v_autonomy=[100,150,200]
    #trips_autonomy=40
    range_trips=[20,40,60]

    #vehicles parameters
    battery_capacity=24 #kWh
    min_charging_th=0.2
    max_charging_th=0.9
    battery_capacity_reduced=battery_capacity*(max_charging_th-min_charging_th)
    av_consumption=0.17 #kWh/km 
    dist_autonomy=battery_capacity_reduced/av_consumption #km
    av_trip_length=4 #km
    trips_autonomy=dist_autonomy/av_trip_length
    
    #CS parameters
    outlet_power=20 #kW
    outlet_rate=outlet_power/battery_capacity_reduced #hourly rate of charging operation
    outlet_per_stations=2
    
    ## ASSIGN CHARGING STATIONS TO ZONE
    zones_with_charging_stations=np.sort(np.random.choice(n_zones,n_charging_stations,replace=False))

    #generate vector with number of servers per each zone
    n_servers=np.ones(n_zones+n_charging_stations)
    n_servers[n_zones:n_zones+n_charging_stations]=np.ones(n_charging_stations)*outlet_per_stations
    
    ## BUILD SERVICE RATES VECTOR per zone
    service_rates=np.ones(n_zones+n_charging_stations)
    service_rates[0:n_zones]=zone_rates_from_file
    #service_rates[0:n_zones]=np.random.randint(low=1, high=5, size=n_zones)
    #service rate of the single charging outlet
    service_rates[n_zones:n_zones+n_charging_stations]=(np.ones(n_charging_stations)*outlet_rate)
    #service_rates[n_zones:n_zones+n_charging_stations]=np.random.randint(low=1,high=3,size=n_charging_stations)/50
    #service_rates=np.array([10,6])
    #print("\nService rate per zone: ",service_rates)

    #Compute number of possible states
    #n_states=scipy.special.binom(n_zones+n_vehicles-1, n_vehicles)
    #print("State space dimension: ", n_states)

    ## BUILD OD MATRIX - True if cycles are admitted
    #OD_matrix=build_OD(n_zones, False)
    OD_matrix=OD_from_file
    #OD_matrix=np.array([[0, 1],[0.5, 0.5]])
    #print("OD:\n", OD_matrix)

    #Check on eigenvalues of OD matrix 
    #print("Eigenvalues: ", np.linalg.eig(OD_matrix)[0])

    if print_input_data:
        print_data(n_zones, n_vehicles, trips_autonomy, dist_autonomy, outlet_rate, zones_with_charging_stations, outlet_per_stations)

    #ax=sns.heatmap(dist_matrix[0:n_zones,0:n_zones], linewidth=0.5)

    ##DETERMINE FLOWS IN CS AS FUNCTION OF: (need mva at each iteration)
    #   TOTAL THROUGHPUT AND TRIPS AUTONOMY 
    #   TOTAL DISTANCE AND VEHICLES AUTONOMY (set trips_autonomy=None)
    """
    city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput=steady_state_analysis(n_zones, n_charging_stations,  n_vehicles, service_rates, n_servers, OD_matrix, zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, False, False)
   
    if city_grid is not None:
        plot_data_on_grid(city_grid_data, city_grid_CS_data)

    waiting_p=compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, rho, throughput_vector)
    city_grid_CS_data['waiting_p']=waiting_p
    if print_results:
        print_stat(flows, av_vehicles, av_delay, av_waiting, ov_throughput, throughput_vector, rho, waiting_p)
        print(np.sum(throughput_vector))
    
    if plot:
        plot_results(flows, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, range_trips, av_vehicles, waiting_p, unsatisfied_demand_per_zone, zones_id_dict, zones_with_charging_stations, city_grid, False, False)
        #bar chart with zones parameters (set to None to avoid plotting)
        plot_bar_per_zone(n_zones, n_charging_stations, zones_with_charging_stations, av_vehicles, av_delay, av_waiting, throughput_vector, rho, unsatisfied_demand_per_zone, lost_requests_per_zone, zones_id_dict)
    city_grid_data['departure_per_zone']=list(tot_departure_per_zone)
    city_grid_data['arrival_per_zone']=list(tot_arrival_per_zone)
    #print(city_grid_data)
    """
    
    #TRY CUSTOM PLACEMENT FOR CHARGING STATIONS
    trips_data=pd.DataFrame(list(tot_departure_per_zone),index=city_grid.zone_id, columns=['departure_per_zone'])
    trips_data['arrival_per_zone']=list(tot_arrival_per_zone)
    #order_by: ['departure_per_zone','arrival_per_zone']
    _zones_with_charging_stations=place_CS_by(trips_data, zones_id_dict, n_charging_stations, order_by='arrival_per_zone', ascending_order=False)
    print("Zones with CS: ",_zones_with_charging_stations)
    #plot_city_zones(city_grid,_zones_with_charging_stations,True)
    
    #without relocation for charging
    case=f'{case_time}_no_reloc_{n_charging_stations}CS'
    print(f"\nCASE: {case}")
    city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput=steady_state_analysis(n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, OD_matrix, _zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, False, False)
    if not city_grid_data.empty:
        plot_data_on_grid(city_grid_data, city_grid_CS_data, ov_throughput)
        waiting_p=compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, city_grid_CS_data)
        city_grid_CS_data['waiting_p']=waiting_p
        avg_CS_utilization=city_grid_CS_data['utilization'].mean()
        print("Average CS utilization: ", avg_CS_utilization*100)
        print("Max CS utilization: ", city_grid_CS_data['utilization'].max()*100)

        #print_stat(city_grid_data, city_grid_CS_data, ov_throughput)
        plot_results(city_grid_data, city_grid_CS_data, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, zones_id_dict, _zones_with_charging_stations)
        plot_bar_per_zone(n_zones, n_charging_stations, _zones_with_charging_stations, city_grid_data, city_grid_CS_data, zones_id_dict)
    else:
        print("No convergence of flows!")
    
    #with relocation for charging (uniform to CS)
    case=f'{case_time}_u_reloc_{n_charging_stations}CS'
    print(f"\nCASE: {case}")
    if not city_grid_data.empty:
        city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput=steady_state_analysis(n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, OD_matrix, _zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, True, False)
        plot_data_on_grid(city_grid_data, city_grid_CS_data, ov_throughput)
        waiting_p=compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, city_grid_CS_data)
        city_grid_CS_data['waiting_p']=waiting_p
        avg_CS_utilization=city_grid_CS_data['utilization'].mean()
        print("Average CS utilization: ", avg_CS_utilization*100)
        print("Max CS utilization: ", city_grid_CS_data['utilization'].max()*100)
        #print_stat(city_grid_data, city_grid_CS_data, ov_throughput)
        plot_results(city_grid_data, city_grid_CS_data, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, zones_id_dict, _zones_with_charging_stations)
        plot_bar_per_zone(n_zones, n_charging_stations, _zones_with_charging_stations, city_grid_data, city_grid_CS_data, zones_id_dict)
    else:
        print("No convergence of flows!")
    #with relocation for charging (nearest CS)
    case=f'{case_time}_closest_reloc_{n_charging_stations}CS'
    print(f"\nCASE: {case}")
    city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput=steady_state_analysis(n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, OD_matrix, _zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, True, True)
    if not city_grid_data.empty:
        plot_data_on_grid(city_grid_data, city_grid_CS_data, ov_throughput)
        waiting_p=compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, city_grid_CS_data)
        city_grid_CS_data['waiting_p']=waiting_p
        avg_CS_utilization=city_grid_CS_data['utilization'].mean()
        print("Average CS utilization: ", avg_CS_utilization*100)
        print("Max CS utilization: ", city_grid_CS_data['utilization'].max()*100)

        #print_stat(city_grid_data, city_grid_CS_data, ov_throughput)
        plot_results(city_grid_data, city_grid_CS_data, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, zones_id_dict, _zones_with_charging_stations)
        plot_bar_per_zone(n_zones, n_charging_stations, _zones_with_charging_stations, city_grid_data, city_grid_CS_data, zones_id_dict)
    else:
        print("No convergence of flows!")
    
    
    #CONNCENTRATED CS
    n_charging_stations=2
    outlet_per_stations=10
    #generate vector with number of servers per each zone
    n_servers=np.ones(n_zones+n_charging_stations)
    n_servers[n_zones:n_zones+n_charging_stations]=np.ones(n_charging_stations)*outlet_per_stations
    ## BUILD SERVICE RATES VECTOR per zone
    service_rates=np.ones(n_zones+n_charging_stations)
    service_rates[0:n_zones]=zone_rates_from_file
    service_rates[n_zones:n_zones+n_charging_stations]=(np.ones(n_charging_stations)*outlet_rate)

    _zones_with_charging_stations=place_CS_by(trips_data, zones_id_dict, n_charging_stations, order_by='arrival_per_zone', ascending_order=False)
    print("Zones with CS: ",_zones_with_charging_stations)
    #without reloc4charg
    case=f'{case_time}_no_reloc_{n_charging_stations}CS'
    print(f"\nCASE: {case}")
    city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput=steady_state_analysis(n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, OD_matrix, _zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, False, False)
    if not city_grid_data.empty:
        plot_data_on_grid(city_grid_data, city_grid_CS_data, ov_throughput)
        waiting_p=compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, city_grid_CS_data)
        city_grid_CS_data['waiting_p']=waiting_p
        #print_stat(city_grid_data, city_grid_CS_data, ov_throughput)
        plot_results(city_grid_data, city_grid_CS_data, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, zones_id_dict, _zones_with_charging_stations)
        plot_bar_per_zone(n_zones, n_charging_stations, _zones_with_charging_stations, city_grid_data, city_grid_CS_data, zones_id_dict)
    else:
        print("No convergence of flows!")
    #with relocation for charging (uniform to CS)
    case=f'{case_time}_u_reloc_{n_charging_stations}CS'
    print(f"\nCASE: {case}")
    city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput=steady_state_analysis(n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, OD_matrix, _zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, True, False)
    if not city_grid_data.empty:
        plot_data_on_grid(city_grid_data, city_grid_CS_data, ov_throughput)
        waiting_p=compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, city_grid_CS_data)
        city_grid_CS_data['waiting_p']=waiting_p
        #print_stat(city_grid_data, city_grid_CS_data, ov_throughput)
        plot_results(city_grid_data, city_grid_CS_data, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, zones_id_dict, _zones_with_charging_stations)
        plot_bar_per_zone(n_zones, n_charging_stations, _zones_with_charging_stations, city_grid_data, city_grid_CS_data, zones_id_dict)
    else:
        print("No convergence of flows!")
    #with relocation for charging (nearest CS)
    case=f'{case_time}_closest_reloc_{n_charging_stations}CS'
    print(f"\nCASE: {case}")
    city_grid_data, city_grid_CS_data, aug_OD_matrix, ov_throughput=steady_state_analysis(n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, OD_matrix, _zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, True, True)
    if not city_grid_data.empty:
        plot_data_on_grid(city_grid_data, city_grid_CS_data, ov_throughput)
        waiting_p=compute_waiting_probability(n_zones, n_charging_stations, n_servers, service_rates, city_grid_CS_data)
        city_grid_CS_data['waiting_p']=waiting_p
        #print_stat(city_grid_data, city_grid_CS_data, ov_throughput)
        plot_results(city_grid_data, city_grid_CS_data, aug_OD_matrix, range_vehicles, n_zones, n_charging_stations, n_vehicles, service_rates, n_servers, zones_id_dict, _zones_with_charging_stations)
        plot_bar_per_zone(n_zones, n_charging_stations, _zones_with_charging_stations, city_grid_data, city_grid_CS_data, zones_id_dict)
    else:
        print("No convergence of flows!")
    
    """
    #iterate for global indexes
    od_hour_range=list(np.arange(0,24))
    #od_hour_range.append(np.arange(0,24))
    #print(od_hour_range)
    ov_throughput_list=[]
    ov_ud_list=[]
    
    for hour in od_hour_range:
        city_grid, n_zones, zones_id_dict, OD_from_file, zone_rates_from_file, tot_departure_per_zone, tot_arrival_per_zone=get_hour_OD("new_grid.pickle","trips_with_zone_id.csv",hour, True)
        #redifine params may change n_zones
        #_zones_with_charging_stations=place_CS_by(city_grid_data, zones_id_dict, n_charging_stations, order_by='arrival_per_zone', ascending_order=False)
        OD_matrix=OD_from_file
        service_rates=np.ones(n_zones+n_charging_stations)
        service_rates[0:n_zones]=zone_rates_from_file
        n_servers=np.ones(n_zones+n_charging_stations)
        n_servers[n_zones:n_zones+n_charging_stations]=np.ones(n_charging_stations)*outlet_per_stations
        #analysis
        flows, aug_OD_matrix, av_vehicles, av_delay, ov_throughput, throughput_vector, av_waiting, rho, unsatisfied_demand_per_zone, lost_requests_per_zone=steady_state_analysis(n_zones, n_charging_stations, OD_matrix, _zones_with_charging_stations, trips_autonomy, city_grid, zones_id_dict, False, False)
        ov_throughput_list.append(ov_throughput)
        #print("ud:", unsatisfied_demand_per_zone)
        ov_ud_list.append(np.sum((np.multiply(unsatisfied_demand_per_zone[0:n_zones],service_rates[0:n_zones]))/np.sum(service_rates[0:n_zones])))
        #plot_data_on_grid(flows, throughput_vector, av_vehicles, av_delay, rho, unsatisfied_demand_per_zone, _zones_with_charging_stations)
    ov_ud_list=[i*100 for i in ov_ud_list]
    
    print(ov_throughput_list)
    print(ov_ud_list)
    plot_gloal_indexes(ov_throughput_list, ov_ud_list)
    """
    if n_zones+n_charging_stations<10:
        flows=list(city_grid_data['flows']).append(i for i in city_grid_CS_data['flows'])
        flows=np.array(flows)
        product_form_distribution(n_zones, n_charging_stations, n_vehicles, flows, service_rates, n_servers)
    
    #plt.show()
    plt.close('all')

    """
    #AVG # VEHICLES IN CS AS FUNCTION OF TRIPS AUTONOMY
    fig5, ax5=plt.subplots()
    for _trips_autonomy in range_trips:
        _city_grid_data, _city_grid_CS_data, _aug_OD_matrix, _ov_throughput=steady_state_analysis(n_zones, n_charging_stations, OD_matrix, zones_with_charging_stations, _trips_autonomy, city_grid, zones_id_dict, reloc4charg, closest_CS)
        #_flows, _aug_OD=total_fluxes(n_zones,n_charging_stations,OD_matrix,zones_with_charging_stations,_trips_autonomy)
        #_av_vehicles, _av_delay, _ov_throughput=MS_MVA(n_zones+n_charging_stations, n_vehicles, service_rates, _flows, n_servers)
        ax5.bar(np.arange(n_charging_stations),_city_grid_CS_data['av_vehicles'], label=f'trips autonomy: {_trips_autonomy}', alpha=_trips_autonomy/(max(range_trips)+10))
    ax5.set_title("Average number of vehicles in Charging station")
    ax5.set_xlabel("Charging  stations")
    ax5.set_ylabel("Vehicles")
    ax5.set_xticks(list(np.arange(n_charging_stations)))
    ax5.set_xticklabels(cs_ticks, rotation=45)
    ax5.grid()
    ax5.legend()
    #ax5.scatter(np.arange(n_charging_stations),n_servers[n_zones:n_zones+n_charging_stations], label=f'n servers', color='red',marker='.')
    #plt.show()
    """
    """
    ## CUST EXAMPLE
    en_zones=3
    en_charging_stations=0
    en_vehicles=8
    #en_servers=np.ones(en_zones+en_charging_stations)
    en_servers=np.array([1,2,4])
    eservice_rates=np.array([1,1,1/8])
    eOD_matrix=np.array([[0,0.1,0.9],[0.8,0,0.2],[0.7,0.3,0]])
    ezones_with_charging_stations=np.sort(np.random.choice(en_zones,en_charging_stations,replace=False))
    print("zones with cs: ",ezones_with_charging_stations)
    edistance_matrix=build_distance_matrix(en_zones, en_charging_stations, ezones_with_charging_stations)

    eflows, eaug_OD=fluxes(en_zones,en_charging_stations,eOD_matrix,ezones_with_charging_stations,edistance_matrix,service_rates,0)
    
    print("\nRelative flows: ",eflows)
    eav_vehicles, eav_delay, eov_throughput=MS_MVA(en_zones+en_charging_stations, en_vehicles, eservice_rates, eflows, en_servers)
    ethroughput_vector=np.round(eov_throughput*eflows,4)
    eav_waiting=eav_delay-(1/eservice_rates)
    print("\nAverage vehicles vector: ", eav_vehicles)
    print("Average vehicles idle time vector: ", eav_delay)
    print('Average vehicles "waiting" time vector: ', eav_waiting)
    print("Overall throughput (constant for the flow calculation): ", eov_throughput)
    print("Throughputs vector (Real flows): ", ethroughput_vector)
    
    #compute utilization vector (rho) with computed flows and service rates
    erho=np.round(np.divide(ethroughput_vector,np.multiply(eservice_rates,en_servers)),4)
    print("\nrho (utilization) per zone: ", erho)
    
    #Compute unsatisfied demand and mobility requests for mobility zones only
    eunsatisfied_demand_per_zone=((1-erho[0:en_zones]))
    np.set_printoptions(suppress=True)
    print(f"\nPercentage of unsatisfied demand per zone: {eunsatisfied_demand_per_zone*100}%")
    print(f"Average demand lost: {np.round(np.sum(eunsatisfied_demand_per_zone*100)/en_zones,2)}%")

    elost_requests_per_zone=np.round(np.multiply(eunsatisfied_demand_per_zone,eservice_rates[0:en_zones]),4)
    print("Requests lost per unit time per zone: ", elost_requests_per_zone)
    print("Total requests lost per unit time: ",np.round(np.divide(np.sum(elost_requests_per_zone),np.sum(eservice_rates[0:en_zones])),4))
    
    #counter calculation
    eflows=np.multiply(np.array([0.493,0.246,0.985]),np.multiply(eservice_rates,en_servers))/0.4927
    eav_vehicles2, eav_delay2, eov_throughput2=MS_MVA(en_zones+en_charging_stations, en_vehicles, eservice_rates, eflows, en_servers)
    print("Overall throughput2 (constant for the flow calculation): ", eov_throughput2)
    """

    """
    range_v=np.arange(n_zones,600)
    avg_uns_demand_list=[]
    tot_lost_requests_list=[]
    for n_vehicles in range_v:
        av_vehicles, av_delay, ov_throughput=MVA(n_zones, n_vehicles, service_rates, flows)
        throughput_vector=np.round(ov_throughput*flows,4)
        rho=np.round(np.divide(throughput_vector,service_rates),4)
        unsatisfied_demand_per_zone=((1-rho)*100)
        avg_uns_demand_list.append(np.sum(unsatisfied_demand_per_zone)/n_zones)
        lost_requests_per_zone=np.round(np.multiply(unsatisfied_demand_per_zone,service_rates),4)
        tot_lost_requests_list.append(np.sum(lost_requests_per_zone))
       
    
    fig, ax = plt.subplots()
    ax.plot(range_v,avg_uns_demand_list,linewidth=2.0, label='Average unsatisfied demand [%]')
    ax.set_title(f"Unsatisfied mobility demand for {n_zones} zones")
    ax.set_xlabel("Fleet size")
    ax.grid()
    ax.legend()
    #plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(range_v, tot_lost_requests_list,linewidth=2.0, label='Total requests lost per unit time')
    ax2.set_title(f"Lost mobility requests for {n_zones} zones")
    ax2.set_xlabel("Fleet size")
    ax2.grid()
    ax2.legend()
    #plt.show()
    """