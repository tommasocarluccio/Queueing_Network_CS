import numpy as np
import scipy.special
import time
n_zones=3
n_vehicles=4
transition_probability=1
np.random.seed(42)
"""
OD_matrix=np.full((n_zones,n_zones),transition_probability)
for i in range(n_zones):
    OD_matrix[i,i]=0

"""
#define OD_matrix with no cycles and closed network constraints
OD_matrix=np.zeros((n_zones,n_zones))
for i in range(n_zones):
    transitions=np.random.randint(100, size=n_zones-1)
    transitions=np.round(transitions/np.sum(transitions),2)
    pos=0
    for j in range(n_zones):
        if j!=i:
            OD_matrix[i,j]=transitions[pos]
            pos+=1

service_rates=np.random.randint(low=5, high=15, size=n_zones)
print("mu: ",service_rates)
n_states=scipy.special.binom(n_zones+n_vehicles-1, n_vehicles)
print("State space dimension: ", n_states)


print("OD:\n", OD_matrix)
#print(np.linalg.eig(OD_matrix))
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
#flows=flows*100
#flows=np.round(flows,2)
print("flows: ", flows)
#flows equal service rate in queue 1
flows_normalized=flows/flows[0]*service_rates[0]
print("normalized flows: ", flows_normalized)
print("num it: ", num_it)
flows=flows_normalized

rho=np.divide(flows,service_rates)
print("rho: ", rho)
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

#Convolution algorithm for Normalization coefficient v2 (more efficient)
t1=time.time()
g1=np.zeros(n_vehicles+1)
g1[0]=1
for m in range(0,n_zones):
    for n in range(1,n_vehicles+1):
        g1[n]=g1[n]+rho[m]*g1[n-1]
print("Normalization constant: ",g1[n_vehicles])
#print("time: ", time.time()-t1)

#Mean Value Analysis (MVA)
average_vehicles=np.zeros((n_zones,n_vehicles))
average_waiting=np.zeros((n_zones,n_vehicles))
for m in range(1,n_vehicles):
    for n in range(n_zones):
        average_waiting[n,m]=(1+average_vehicles[n,m-1])/(service_rates[n])
    overall_throughput=m/np.sum(np.multiply(average_waiting[:,m],flows))
    average_vehicles[:,m]=np.multiply(flows*overall_throughput,average_waiting[:,m])
print("Avergae vehicles vector: ", average_vehicles[:,-1])
print("Avergae waiting time vector: ", average_waiting[:,-1])
print("Overall throughputs: ", overall_throughput)
print("Throughputs vector: ", overall_throughput*flows)
