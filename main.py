from cmath import nan
import numpy as np
import scipy.special
import itertools
import time
import matplotlib.pyplot as plt

"""
OD_matrix=np.full((n_zones,n_zones),transition_probability)
for i in range(n_zones):
    OD_matrix[i,i]=0

"""
def build_OD(n_zones, cycles=False):
    #define OD_matrix with closed network constraint
    seed=0
    OD_matrix=np.zeros((n_zones,n_zones))
    valid=False
    while not valid:
        np.random.seed(seed)
        for i in range(n_zones):
            if not cycles: #with no cycles
                transitions=np.random.randint(1000, size=n_zones-1) #random transitions vector
                transitions=np.round(transitions/np.sum(transitions),2) #normalized transitions vector
                pos=0
                for j in range(n_zones): #assign transition probabilities to non diagonal entries
                    if j!=i:
                        OD_matrix[i,j]=transitions[pos]
                        pos+=1
            else: #with cycles
                #assign normalized transitions vector to OD row
                transitions=np.random.randint(100, size=n_zones)
                transitions=np.round(transitions/np.sum(transitions),2)
                OD_matrix[i]=transitions
        if np.real(np.linalg.eig(OD_matrix)[0][0])==1: #check for ergodicity of the matrix
            valid=True
            print("OD seed: ",seed)
            return OD_matrix
        else:
            seed+=1

def compute_fluxes(n_zones, OD_matrix, service_rates, normalized=True):
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
    if normalized: #flows equal service rate in queue 1
        flows_normalized=flows/flows[0]*service_rates[0]
        flows=flows_normalized
    return flows

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
    average_vehicles=np.zeros((n_zones,n_vehicles)) #average number of vehicles per zone
    average_waiting=np.zeros((n_zones,n_vehicles)) #average "waiting time" of vehicles per zone
    for m in range(1,n_vehicles):
        for n in range(n_zones):
            average_waiting[n,m]=(1+average_vehicles[n,m-1])/(service_rates[n])
        overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
        average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
    return average_vehicles[:,-1], average_waiting[:,-1], overall_throughput

def compute_generic_pi(vehicles_vector, rho, normalization_constant):
    pi=np.prod(rho**vehicles_vector)/normalization_constant
    return np.round(pi,4)

def compute_pi0(n_zones, n_vehicles, rho, normalization_constant):
    #Find pi for states where at least a queue is empty 
    tot_pi_0=0
    empty_queues=[]
    v_vector=range(0,n_vehicles+1)
    #All possible arrangements of n_vehicles in n_zones
    for vehicle_per_zone in itertools.product(v_vector, repeat=n_zones):
        if np.sum(vehicle_per_zone)==n_vehicles and (np.array(vehicle_per_zone)==0).any(): #states when at least one queue is empty
            pi=np.round(np.prod(rho**(np.array(vehicle_per_zone)))/normalization_constant,4) #probability of state with product form
            empty_queues.append((list(vehicle_per_zone),pi))
            tot_pi_0+=pi
    #print("Probabilities for unsatisfied demand states:\n",empty_queues)
    #print("Overall probability of unsatisfied demand states: ",np.round(tot_pi_0,4))
    return np.round(tot_pi_0,4)

def compute_pi0_2(n_zones, n_vehicles, rho, normalization_constant):
        tot_pi_0=0
        tot_vehicles_lost=0
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
                        vehicles_lost=np.round((np.sum(partial_mu))*pi,4)
                        pos=0
                        v_complete=[]
                        for i in range(id.size):
                            if id[i]:
                                v_complete.append(vehicle_per_zone[pos])
                                pos+=1
                            else:
                                v_complete.append(0)
                        empty_queues.append((v_complete,pi,vehicles_lost))
                        tot_vehicles_lost+=vehicles_lost
                        tot_pi_0+=pi
        print("empty queues: ", empty_queues)
        #print("Tot pi0: ", np.round(tot_pi_0,4))
        return np.round(tot_pi_0,4), np.round(tot_vehicles_lost,4)

def plot_pi0(n_zones, n_vehicles_max, rho):
        tot_pi0_vector=[]
        vehicles_range=range(1,n_vehicles_max)
        for n_vehicles in vehicles_range:
            normalization_constant=compute_normalization_constant(n_zones, n_vehicles, rho)
            tot_pi0_vector.append(compute_pi0_2(n_zones, n_vehicles, rho, normalization_constant))
        
        fig, ax = plt.subplots()
        ax.plot(vehicles_range, tot_pi0_vector, linewidth=2.0)
        ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
        ax.set_title(f"Total pi0 as function of vehicles number for {n_zones} zones")
        ax.set_xlabel("Number of vehicles")
        ax.set_ylabel("Total pi0")
        ax.grid()
        plt.show()


if __name__=="__main__":
    np.random.seed(42)
    n_zones=3
    n_vehicles=10
    #transition_probability=1
    #Generate vector of service rates per zone
    service_rates=np.random.randint(low=5, high=15, size=n_zones)
    print("mu: ",service_rates)
    #Compute number of possible states
    n_states=scipy.special.binom(n_zones+n_vehicles-1, n_vehicles)
    print("State space dimension: ", n_states)

    OD_matrix=build_OD(n_zones, False)
    print("OD:\n", OD_matrix)
    #Check on eigenvalues of OD matrix for ergodicity
    #print("Eigenvalues: ", np.linalg.eig(OD_matrix)[0])

    flows=compute_fluxes(n_zones, OD_matrix, service_rates, True)
    print("flows: ", flows)
    #compute utilization vector (rho) with computed flows and service rates
    rho=np.divide(flows,service_rates)
    print("rho: ", rho)

    normalization_constant=compute_normalization_constant(n_zones, n_vehicles, rho)
    av_vehicles, av_waiting, ov_throughput=MVA(n_zones, n_vehicles, service_rates, flows)

    
    print("Normalization constant: ", normalization_constant)
    print("Avergae vehicles vector: ", av_vehicles)
    print("Avergae waiting time vector: ", av_waiting)
    print("Overall throughput: ", ov_throughput)
    print("Throughputs vector: ", ov_throughput*flows)
    """
    t=time.time()
    tot_pi0=compute_pi0(n_zones, n_vehicles, rho, normalization_constant)
    print("Total pi0: ", tot_pi0)
    print("t1: ", time.time()-t)
    """
    #compute pi(0,4,6)
    vehicles_vector=np.array([0,4,6])
    pi=compute_generic_pi(vehicles_vector, rho, normalization_constant)
    print(f"Probability of {vehicles_vector}: {pi}")

    t=time.time()
    tot_pi0_2, tot_vehicles_lost=compute_pi0_2(n_zones, n_vehicles, rho, normalization_constant)
    print("Total pi0: ", tot_pi0_2)
    print("Total vehicles lost per time unit: ", tot_vehicles_lost)
    print("t2: ", time.time()-t)

    #plot_pi0(n_zones,50,rho)

