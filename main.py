import numpy as np
import scipy.special
import time

"""
OD_matrix=np.full((n_zones,n_zones),transition_probability)
for i in range(n_zones):
    OD_matrix[i,i]=0

"""
def build_OD(n_zones, cycles=False):
    #define OD_matrix with closed network constraints
    OD_matrix=np.zeros((n_zones,n_zones))
    for i in range(n_zones):
        if not cycles: #no cycles
            transitions=np.random.randint(100, size=n_zones-1)
            transitions=np.round(transitions/np.sum(transitions),2)
            pos=0
            for j in range(n_zones):
                if j!=i:
                    OD_matrix[i,j]=transitions[pos]
                    pos+=1
        else: #with cycles
            transitions=np.random.randint(100, size=n_zones)
            transitions=np.round(transitions/np.sum(transitions),2)
            OD_matrix[i]=transitions
    return OD_matrix

def compute_fluxes(n_zones, OD_matrix, service_rates, normalized=True):
    lambda_vec=np.ones((n_zones))
    num_it=0
    iterate=True
    while iterate:
        flows=np.dot(lambda_vec,OD_matrix)
        if (abs(flows-lambda_vec)>10e-8).any():
            lambda_vec=flows
            num_it+=1
        else:
            iterate=False
    #print("num it: ", num_it)
    if normalized: #flows equal service rate in queue 1
        flows_normalized=flows/flows[0]*service_rates[0]
        flows=flows_normalized
    return flows

"""
#Convolution algorithm for Normalization coefficient
t0=time.time()
g=np.zeros((n_vehicles+1,n_zones))
for i in range(n_zones):
    g[0,i]=1
for i in range(n_vehicles+1):
    g[i,0]=rho[0]**i
for i in range(1,n_vehicles+1):
    for j in range(1,n_zones):
        g[i,j]=g[i,j-1]+rho[j]*g[i-1,j]
#print(g)
G=g[n_vehicles,n_zones-1]
#print(G)
#print("time: ", time.time()-t0)
"""
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
    average_vehicles=np.zeros((n_zones,n_vehicles))
    average_waiting=np.zeros((n_zones,n_vehicles))
    for m in range(1,n_vehicles):
        for n in range(n_zones):
            average_waiting[n,m]=(1+average_vehicles[n,m-1])/(service_rates[n])
        overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
        average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
    return average_vehicles[:,-1], average_waiting[:,-1], overall_throughput

if __name__=="__main__":
    n_zones=3
    n_vehicles=4
    transition_probability=1
    np.random.seed(35)
    service_rates=np.random.randint(low=5, high=15, size=n_zones)
    print("mu: ",service_rates)
    n_states=scipy.special.binom(n_zones+n_vehicles-1, n_vehicles)
    print("State space dimension: ", n_states)

    OD_matrix=build_OD(n_zones, False)
    print("OD:\n", OD_matrix)
    print("Eigenvalues: ", np.linalg.eig(OD_matrix)[0])

    flows=compute_fluxes(n_zones, OD_matrix, service_rates, True)
    rho=np.divide(flows,service_rates)
    print("rho: ", rho)

    normalization_constant=compute_normalization_constant(n_zones, n_vehicles, rho)
    av_vehicles, av_waiting, ov_throughput=MVA(n_zones, n_vehicles, service_rates, flows)

    print("flows: ", flows)
    print("Normalization constant: ", normalization_constant)
    print("Avergae vehicles vector: ", av_vehicles)
    print("Avergae waiting time vector: ", av_waiting)
    print("Overall throughput: ", ov_throughput)
    print("Throughputs vector: ", ov_throughput*flows)