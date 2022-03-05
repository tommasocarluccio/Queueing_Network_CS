from cmath import nan
import numpy as np
import scipy.linalg as la
import scipy.special
import itertools
import time
import matplotlib.pyplot as plt
import pandas as pd

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
    
    for i in range(n_charging_stations):
        matrix[n_zones+i,:]=matrix[zones_with_charging_stations[i],:]
        matrix[:,n_zones+i]=matrix[:,zones_with_charging_stations[i]]
    
    return np.round(matrix,4)

def compute_charging_rates(n_zones, fluxes, OD_matrix, dist_matrix, vehicles_autonomy):
    dist_vector=[]
    rel_flows=np.divide(fluxes,np.sum(fluxes))
    for i in range(n_zones):
        dist_vector.append(np.sum(np.multiply(OD_matrix[i,:],dist_matrix[i,:])))
    charging_rates=np.multiply(dist_vector,rel_flows[0:n_zones])/vehicles_autonomy
    return charging_rates
    
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
    t1=time.time()
    g1=np.zeros(n_vehicles+1)
    g1[0]=1
    for m in range(0,n_zones):
        for n in range(1,n_vehicles+1):
            g1[n]=g1[n]+rho[m]*g1[n-1]
    #print("time: ", time.time()-t1)
    return g1[n_vehicles]
   
def MVA(n_zones, n_vehicles, service_rates, flows):
    #Mean Value Analysis (MVA)
    average_vehicles=np.zeros((n_zones,n_vehicles+1)) #average number of vehicles per zone
    average_waiting=np.zeros((n_zones,n_vehicles+1)) #average "waiting time" of vehicles per zone
    for m in range(1,n_vehicles+1):
        for n in range(n_zones):
            average_waiting[n,m]=(1+average_vehicles[n,m-1])/(service_rates[n])
        overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
        average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
    return np.round(average_vehicles[:,-1],4), np.round(average_waiting[:,-1],4), np.round(overall_throughput,4)

def MS_AMVA(n_zones, n_vehicles, service_rates, flows, n_servers):
    #Mean Value Analysis (MVA)
    average_vehicles=np.zeros((n_zones,n_vehicles+1)) #average number of vehicles per zone
    average_waiting=np.zeros((n_zones,n_vehicles+1)) #average "waiting time" of vehicles per zone
    for m in range(1,n_vehicles+1):
        for n in range(n_zones):
            correction_factor=(1/n_servers[n])*(flows[n]**((n_servers[n]**0.676)-1))
            average_waiting[n,m]=(1+correction_factor*average_vehicles[n,m-1])/(service_rates[n])
        overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
        average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
    return np.round(average_vehicles[:,-1],4), np.round(average_waiting[:,-1],4), np.round(overall_throughput,4)


def MS_MVA(n_zones, n_vehicles, service_rates, flows, n_servers):
    #Multi servers Mean Value Analysis (MS-MVA)
    average_vehicles=np.zeros((n_zones,n_vehicles+1)) #average number of vehicles per zone
    average_waiting=np.zeros((n_zones,n_vehicles+1)) #average "waiting time" of vehicles per zone
    max_ns=int(np.max(n_servers))
    p=np.zeros((max_ns,n_vehicles+1))
    p[0,0]=1
    for m in range(1,n_vehicles+1):
        for n in range(n_zones):
            ns=int(n_servers[n])
            correction_factor=0
            if ns!=1: 
                """
                for j in range(ns-1):
                    #check correction factor
                    correction_factor+=(ns-j)*p[j,m-1]
                """
                for j in range(1,ns):
                    correction_factor+=(ns-j)*p[j-1,m-1]
            average_waiting[n,m]=(1+average_vehicles[n,m-1]+correction_factor)/(service_rates[n]*ns)
        overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
        average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
        for n in range (n_zones):
            ns=int(n_servers[n])
            if ns!=1:
                su=0
                for j in range(1,ns):
                    p[j,m]=(1/j)*(flows[n]/(service_rates[n]*ns))*overall_throughput*p[j-1,m-1]
                    su+=(ns-j)*p[j,m]
                p[0,m]=1-(1/ns)*(overall_throughput*flows[n]/(service_rates[n]*ns)+su)
    #print("PP:\n",p)
    return np.round(average_vehicles[:,-1],4), np.round(average_waiting[:,-1],4), np.round(overall_throughput,4)

def compute_generic_pi(vehicles_vector, rho, normalization_constant):
    pi=np.prod(rho**vehicles_vector)/normalization_constant
    return np.round(pi,4)

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
        plt.show()

def plot_bar_per_zone(n_zones, n_charging_zones, avg_vehicles, idle_times, throughputs, utilization, un_demand, lost_requests):
    if avg_vehicles is not None:
        fig11, ax11 = plt.subplots()
        ax11.bar(np.arange(1,n_zones+n_charging_zones+1),avg_vehicles)
        ax11.set_title(f"Average vehicles per zones")
        ax11.set_xlabel("Zone")
        ax11.set_ylabel("Vehicles")
        ax11.grid()
        plt.show()
    if idle_times is not None:
        fig12, ax12 = plt.subplots()
        ax12.bar(np.arange(1,n_zones+n_charging_zones+1),idle_times)
        ax12.set_title(f"Average vehicles idle times per zones")
        ax12.set_xlabel("Zone")
        ax12.set_ylabel("Idle time [s]")
        ax12.grid()
        plt.show()
    if throughputs is not None:
        fig13, ax13 = plt.subplots()
        ax13.bar(np.arange(1,n_zones+n_charging_zones+1),throughputs)
        ax13.set_title(f"Vehicles througput per zones")
        ax13.set_xlabel("Zone")
        ax13.set_ylabel("Throughput")
        ax13.grid()
        plt.show()
    if utilization is not None:
        fig14, ax14 = plt.subplots()
        ax14.bar(np.arange(1,n_zones+n_charging_zones+1),utilization)
        ax14.set_title(f"Utilization per zones")
        ax14.set_xlabel("Zone")
        ax14.set_ylabel("Utilization [%]")
        ax14.grid()
        plt.show()
    if un_demand is not None:
        fig15, ax15 = plt.subplots()
        ax15.bar(np.arange(1,n_zones+1),un_demand)
        ax15.set_title(f"Unsatisfied mobility demand per zones")
        ax15.set_xlabel("Zone")
        ax15.set_ylabel("Unsatisfied demand [%]")
        ax15.grid()
        plt.show()
    if lost_requests is not None:
        fig16, ax16 = plt.subplots()
        ax16.bar(np.arange(1,n_zones+1),lost_requests)
        ax16.set_title(f"Lost mobility requests per zones")
        ax16.set_xlabel("Zone")
        ax16.set_ylabel("Lost requests")
        ax16.grid()
        plt.show()

def fluxes(n_zones, n_charging_stations, OD_matrix, zones_with_charging_stations, dist_matrix, service_rates, vehicles_autonomy):
    np.set_printoptions(suppress=True)
    aug_matrix=np.zeros((n_zones+n_charging_stations,n_zones+n_charging_stations))
    aug_matrix[0:n_zones,0:n_zones]=OD_matrix
    for j in range(n_charging_stations):
        aug_matrix[n_zones+j][zones_with_charging_stations[j]]=1
    #print("\nAUG:\n", aug_matrix)

    lambda_vec=np.ones((n_zones+n_charging_stations)) #initialize vector of flows
    #print("lambda init:\n",lambda_vec)
    num_it=0
    """
    #FIRST ITERATION WITH FIXED ALPHA
    alpha=1
    flows=np.dot(lambda_vec,aug_matrix) #compute vector of flows
    if n_charging_stations>0:
            for i in range(n_charging_stations):
                correction_coefficient=alpha*flows[zones_with_charging_stations[i]]*(np.sum(np.delete(flows,n_zones+i)/(n_zones+n_charging_stations-1)))
                aug_matrix[0:n_zones,n_zones+i]=flows[0:n_zones]*correction_coefficient
            aug_matrix=aug_matrix/aug_matrix.sum(axis=1)[:,None]
    print("1ST AUG:\n",aug_matrix)
    """
    #ITERATE WITH COMPUTED ALPHA FROM CHARGING RATES
    iterate=True
    while iterate:
        flows=np.dot(lambda_vec,aug_matrix) #compute vector of flows
        #print("\nflooows: ",flows)
        #print("CH rates:\n", charging_rates)
        #print("pAUG:\n",aug_matrix)
        if n_charging_stations>0:
            charging_rates=compute_charging_rates(n_zones, flows, aug_matrix, dist_matrix, vehicles_autonomy)
            for i in range(n_charging_stations):
                #alpha given by charging rate in zone i/service rate in zone i
                alpha=charging_rates[zones_with_charging_stations[i]]/service_rates[zones_with_charging_stations[i]]
                #print("alpha: ",alpha)
                correction_coefficient=alpha*(1/(n_zones+n_charging_stations))*(np.sum(flows)/(n_zones+n_charging_stations))
                aug_matrix[0:n_zones,n_zones+i]=flows[0:n_zones]*correction_coefficient
            aug_matrix=aug_matrix/aug_matrix.sum(axis=1)[:,None]
        #print("AUUUG:\n",aug_matrix)
        #print("flooows:\n", flows)
        if (abs(flows-lambda_vec)>10e-4).any(): #check on convergence
            lambda_vec=flows
            num_it+=1
        else:
            iterate=False
    #print("augmented OD matrix:\n", aug_matrix)
    print("num it: ", num_it)
    return flows, aug_matrix

if __name__=="__main__":
    np.random.seed(42)
    n_zones=20
    n_vehicles=40
    n_charging_stations=3
    outlet_per_stations=3
    vehicles_autonomy=500

    charging_stations=np.zeros(n_zones)
    zones_with_charging_stations=np.sort(np.random.choice(charging_stations.shape[0],n_charging_stations,replace=False))
    print("zones_with_charging_stations", zones_with_charging_stations)
    charging_stations[np.array(zones_with_charging_stations)]=1 #vector has 1 in position of zones with charging station inside, 0 otherwise

    n_servers=list(np.ones(n_zones))
    #n_servers=[1,2,4]
    #n_servers[20]=10
    #n_servers[1]=3
    #n_servers[2]=4
    
    #Generate vector of service rates per zone
    service_rates=np.random.randint(low=1, high=5, size=n_zones)
    #service_rates=np.array([1,2,2])
    print("service rate per zone: ",service_rates)

    #Compute number of possible states
    n_states=scipy.special.binom(n_zones+n_vehicles-1, n_vehicles)
    #print("State space dimension: ", n_states)

    #Build OD matrix - True if cycles are admitted
    OD_matrix=build_OD(n_zones, False)
    #OD_matrix=np.array([[0, 0.1, 0.9],[0.8, 0, 0.2],[0.7, 0.3, 0]])
    #print("OD:\n", OD_matrix)
    #Check on eigenvalues of OD matrix 
    #print("Eigenvalues: ", np.linalg.eig(OD_matrix)[0])

    dist_matrix=build_distance_matrix(n_zones,n_charging_stations,zones_with_charging_stations)
    #print("dist:\n", dist_matrix)

    #flows_plus=fluxes(n_zones,n_charging_stations,OD_matrix,zones_with_charging_stations)
   
    #flows=compute_fluxes(n_zones, OD_matrix)
    #print("relative flow per zone: ", flows)

    flows, aug_OD_matrix=fluxes(n_zones,n_charging_stations,OD_matrix,zones_with_charging_stations, dist_matrix, service_rates, vehicles_autonomy)
    #flows_plus=aug_fluxes(n_zones,n_charging_stations,OD_matrix,zones_with_charging_stations,flows)
    print("!!", flows)
    
    """
    #Mean Value Analysis of the network
    #flows=np.array([0.843,0.843,0.422])
    av_vehicles, av_waiting, ov_throughput=MVA(n_zones, n_vehicles, service_rates, flows)
    throughput_vector=np.round(ov_throughput*flows,4)
    print("Average vehicles vector: ", av_vehicles)
    print("Average vehicles idle time vector: ", av_waiting)
    print("Overall throughput (constant for the flow calculation): ", ov_throughput)
    print("Throughputs vector (Real flows): ", throughput_vector)

    #compute utilization vector (rho) with computed flows and service rates
    rho=np.round(np.divide(throughput_vector,service_rates),4)
    print("rho (utilization) per zone: ", rho)

    unsatisfied_demand_per_zone=((1-rho))
    np.set_printoptions(suppress=True)
    print(f"Percentage of unsatisfied demand per zone: {unsatisfied_demand_per_zone*100}%")
    print(f"Average demand lost: {np.round(np.sum(unsatisfied_demand_per_zone*100)/n_zones,2)}%")

    lost_requests_per_zone=np.round(np.multiply(unsatisfied_demand_per_zone,service_rates),4)
    print("Requests lost per unit time per zone: ", lost_requests_per_zone)
    print("Total requests lost: ",np.round(np.sum(lost_requests_per_zone),4))
    """
    
    charging_rates=compute_charging_rates(n_zones,flows,aug_OD_matrix,dist_matrix,vehicles_autonomy)
    service_rates_aug=list(service_rates)
    for i in zones_with_charging_stations:
        n_servers.append(outlet_per_stations)
        service_rates_aug.append(charging_rates[i]/outlet_per_stations)
    n_servers=np.array(n_servers)
    service_rates_aug=np.array(service_rates_aug)
    print("\nNew service rates: ",service_rates_aug)
    print("Servers per zone:", n_servers)
    
    #Multi-Server Mean Value Analysis of the network
    av_vehicles, av_waiting, ov_throughput=MS_MVA(n_zones+n_charging_stations, n_vehicles, service_rates_aug, flows, n_servers)
    throughput_vector=np.round(ov_throughput*flows,4)
    print("\nAverage vehicles vector: ", av_vehicles)
    print("Average vehicles idle time vector: ", av_waiting)
    print("Overall throughput (constant for the flow calculation): ", ov_throughput)
    print("Throughputs vector (Real flows): ", throughput_vector)
    
    #compute utilization vector (rho) with computed flows and service rates
    rho=np.round(np.divide(throughput_vector,np.multiply(service_rates_aug,n_servers)),4)
    print("rho (utilization) per zone: ", rho)

    unsatisfied_demand_per_zone=((1-rho[0:n_zones]))
    np.set_printoptions(suppress=True)
    print(f"Percentage of unsatisfied demand per zone: {unsatisfied_demand_per_zone*100}%")
    print(f"Average demand lost: {np.round(np.sum(unsatisfied_demand_per_zone*100)/n_zones,2)}%")

    lost_requests_per_zone=np.round(np.multiply(unsatisfied_demand_per_zone,service_rates),4)
    print("Requests lost per unit time per zone: ", lost_requests_per_zone)
    print("Total requests lost: ",np.round(np.sum(lost_requests_per_zone),4))

    #convolutional algorithm for the normalization constant (used for distribution calculations)
    normalization_constant=compute_normalization_constant(n_zones, n_vehicles, rho)
    #print("Normalization constant: ", normalization_constant)

    """
    #Distribution calculation methods (recursive and combinatorial)
    t1=time.time()
    counters = [0] * n_zones
    state_list = list()
    find_states(counters, n_vehicles, n_zones, 0, state_list)
    #print("States: ", state_list)
    empty_queues, tot_pi0, tot_requests_lost=compute_pi0_rec(state_list, rho, normalization_constant, service_rates)
    print("t rec: ", time.time()-t1)
    print(tot_pi0)
    t2=time.time()
    empty_queues, tot_pi0, tot_requests_lost=compute_pi0_comb(n_zones, n_vehicles, rho, normalization_constant)
    print("t comb: ", time.time()-t2)
    print(tot_pi0)
    #compute pi(0,4,6)
    #vehicles_vector=np.array([0,4,3,3])
    #pi=compute_generic_pi(vehicles_vector, rho, normalization_constant)
    #print(f"Probability of {vehicles_vector}: {pi}") 
    #plot_pi0(n_zones,50,rho)
    """

    
    #cumulative distribution histogram of unsatisfied demand
    fig3, ax3 = plt.subplots()
    ax3.hist(unsatisfied_demand_per_zone, bins=30, cumulative=True, density=True, histtype="stepfilled")
    ax3.set_title(f"Cumulative distribution of unsatisfied mobility demand per zones")
    ax3.set_xlabel("Unsatisfied demand")
    ax3.grid()
    plt.show()

    #bar chart with zones parameters (set to None to avoid plotting)
    #plot_bar_per_zone(n_zones, n_charging_stations, av_vehicles, av_waiting, throughput_vector, rho, unsatisfied_demand_per_zone, lost_requests_per_zone)

    #print(MS_AMVA2(n_zones, n_vehicles, service_rates, flows, n_servers, 45, 0.7))
   
    """
    range_v=np.arange(n_zones,600)
    avg_uns_demand_list=[]
    tot_lost_requests_list=[]
    for n_vehicles in range_v:
        av_vehicles, av_waiting, ov_throughput=MVA(n_zones, n_vehicles, service_rates, flows)
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
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(range_v, tot_lost_requests_list,linewidth=2.0, label='Total requests lost per unit time')
    ax2.set_title(f"Lost mobility requests for {n_zones} zones")
    ax2.set_xlabel("Fleet size")
    ax2.grid()
    ax2.legend()
    plt.show()
    """