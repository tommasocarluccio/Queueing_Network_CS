import math
import matplotlib.pyplot as plt

def compute_pi0(n_vehicles, zone_capacity, rho, n_servers):
    pi0=0
    for i in range(zone_capacity+1):
        if n_servers==1:
            f=1
            for j in range(n_vehicles-i+1,n_vehicles+1):
                f*=j
            pi0+=f*rho**i
        else:
            f1=1
            for j in range(n_servers):
                f1*=(n_vehicles-j)/(j+1)
            f2=1
            for k in range(n_servers,zone_capacity+1):
                f2*=(n_vehicles-k)/n_servers
            pi0+=(rho**i)*f1*f2
    return 1/pi0

def compute_pi_i(n_vehicles, zone_capacity, rho, i, n_servers):
    pi0=compute_pi0(n_vehicles, zone_capacity, rho, n_servers)
    if n_servers==1:
        f=1
        for j in range(n_vehicles-i+1,n_vehicles+1):
            f*=j
        pi=pi0*f*rho**i
    else:
        f1=1
        for j in range(n_servers):
            f1*=(n_vehicles-j)/(j+1)
        f2=1
        for k in range(n_servers,zone_capacity+1):
            f2*=(n_vehicles-k)/n_servers
        pi=pi0*f1*f2*rho**i
    return pi

def compute_pi_noF(n_vehicles, zone_capacity, rho, i):
    rho_noF=rho*n_vehicles
    pi0=(1-rho_noF)/(1-rho_noF**(zone_capacity+1))
    pi=pi0*rho_noF**i
    return pi

def compute_queue_indicators(pi0, n_vehicles, zone_capacity, rho, arrival_rate, n_servers, finite_pop=True):
    avg_vehicles=0
    for i in range(1,zone_capacity+1):
        if finite_pop:
            avg_vehicles+=i*compute_pi_i(n_vehicles,zone_capacity, rho, i, n_servers)
        else:
            avg_vehicles+=i*compute_pi_noF(n_vehicles, zone_capacity, rho, i)
    if finite_pop:
        avg_arrival_rate=0
        loss_prob=0
        for i in range(zone_capacity):
            f=1
            for j in range(n_vehicles-i,n_vehicles+1):
                f*=j
            avg_arrival_rate+=f*arrival_rate*rho**i
            loss_prob+=f*rho**i
        avg_arrival_rate*=pi0
        f=1
        for j in range(n_vehicles-zone_capacity,n_vehicles+1):
            f*=j
        loss_prob=f*rho**zone_capacity/loss_prob
    else:
        avg_arrival_rate=arrival_rate*n_vehicles-arrival_rate*n_vehicles*compute_pi_noF(n_vehicles, zone_capacity,rho,zone_capacity)
        loss_prob=compute_pi_noF(n_vehicles, zone_capacity,rho,zone_capacity)
    
    return avg_vehicles, avg_arrival_rate, loss_prob



if __name__=="__main__":
    n_vehicles=150
    zone_capacity=20
    n_servers=1
    #
    arrival_rate=3 
    service_rate=460
    rho=arrival_rate/service_rate
    pi0=compute_pi0(n_vehicles, zone_capacity, rho, n_servers)
    print("pi0: ", pi0)

    pi3=compute_pi_i(n_vehicles, zone_capacity, rho, 3, n_servers)
    print("pi3: ", pi3)

    avg_vehicles, avg_arrival, loss_prob=compute_queue_indicators(pi0, n_vehicles, zone_capacity, rho, arrival_rate, n_servers, True)
    avg_time=avg_vehicles/avg_arrival

    print("Avergae number of vehicles in the queue: ", avg_vehicles)
    print("Average arrival rate: ", avg_arrival)
    print("Avergae time spent in the queue: ", avg_time)
    print("Loss probabiity due to queue capacity: ", loss_prob)
    
    avg_vehicles_v1=[]
    avg_arrival_v1=[]
    loss_v1=[]
    avg_vehicles_v1_noF=[]
    avg_arrival_v1_noF=[]
    loss_v1_noF=[]
    range1=range(1,100)
    for zone_capacity in range1:
        pi0=compute_pi0(n_vehicles, zone_capacity, rho, n_servers)
        avg_v,avg_a,loss=compute_queue_indicators(pi0, n_vehicles, zone_capacity, rho, arrival_rate, n_servers, True)
        avg_vehicles_v1.append(avg_v)
        avg_arrival_v1.append(avg_a)
        loss_v1.append(loss)
        rho_noF=rho*n_vehicles
        pi0=(1-rho_noF)/(1-rho_noF**(zone_capacity+1))
        avg_v,avg_a,loss=compute_queue_indicators(pi0, n_vehicles, zone_capacity, rho, arrival_rate, n_servers, False)
        avg_vehicles_v1_noF.append(avg_v)
        avg_arrival_v1_noF.append(avg_a)
        loss_v1_noF.append(loss)
        
    fig, ax = plt.subplots()
    ax.plot(range1,avg_vehicles_v1,linewidth=2.0, label='Finite population model')
    ax.plot(range1, avg_vehicles_v1_noF,linewidth=2.0, label='No finite population model')
    #ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
    ax.set_title(f"Average number of vehicles in the queue for fleet size= {n_vehicles}")
    ax.set_xlabel("Zone capacity")
    ax.set_ylabel("Average number of vehicles in the queue")
    ax.grid()
    ax.legend()
    plt.show()

    fig4, ax4 = plt.subplots()
    ax4.plot(range1,avg_arrival_v1,linewidth=2.0, label='Finite population model')
    ax4.plot(range1,avg_arrival_v1_noF,linewidth=2.0, label='No finite population model')
    #ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
    ax4.set_title(f"Average arrival rate in the queue for fleet size= {n_vehicles}")
    ax4.set_xlabel("Zone capacity")
    ax4.set_ylabel("Average arrival rate in the queue")
    ax4.grid()
    ax4.legend()
    plt.show()

    fig5, ax5 = plt.subplots()
    ax5.plot(range1,loss_v1,linewidth=2.0, label='Finite population model')
    ax5.plot(range1,loss_v1_noF,linewidth=2.0, label='No finite population model')
    #ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
    ax5.set_title(f"Loss probability in the queue for fleet size= {n_vehicles}")
    ax5.set_xlabel("Zone capacity")
    ax5.set_ylabel("Loss probability in the queue")
    ax5.grid()
    ax5.legend()
    plt.show()

    zone_capacity=100
    avg_vehicles_v2=[]
    avg_vehicles_v2_noF=[]
    avg_arrival_v2=[]
    avg_arrival_v2_noF=[]
    loss_v2=[]
    loss_v2_noF=[]
    range2=range(zone_capacity+1,500+1)
    for n_vehicles in range2:
        pi0=compute_pi0(n_vehicles, zone_capacity, rho, n_servers)
        avg_v,avg_a,loss=compute_queue_indicators(pi0, n_vehicles, zone_capacity, rho, arrival_rate, n_servers, True)
        avg_vehicles_v2.append(avg_v)
        avg_arrival_v2.append(avg_a)
        loss_v2.append(loss)
        rho_noF=rho*n_vehicles
        pi0=(1-rho_noF)/(1-rho_noF**(zone_capacity+1))
        avg_v,avg_a,loss=compute_queue_indicators(pi0, n_vehicles, zone_capacity, rho, arrival_rate, n_servers, False)
        avg_vehicles_v2_noF.append(avg_v)
        avg_arrival_v2_noF.append(avg_a)
        loss_v2_noF.append(loss)

    fig2, ax2 = plt.subplots()
    ax2.plot(range2,avg_vehicles_v2,linewidth=2.0, label='Finite population model')
    ax2.plot(range2,avg_vehicles_v2_noF,linewidth=2.0, label='No finite population model')
    #ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
    ax2.set_title(f"Average number of vehicles in the queue for zone capacity= {zone_capacity}")
    ax2.set_xlabel("Fleet size")
    ax2.set_ylabel("Average number of vehicles in the queue")
    ax2.grid()
    ax2.legend()
    plt.show()

    fig3, ax3 = plt.subplots()
    ax3.plot(range2,avg_arrival_v2,linewidth=2.0, label='Finite population model')
    ax3.plot(range2,avg_arrival_v2_noF,linewidth=2.0, label='No finite population model')
    #ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
    ax3.set_title(f"Average arrival rate in the queue for zone capacity= {zone_capacity}")
    ax3.set_xlabel("Fleet size")
    ax3.set_ylabel("Average arrival rate in the queue")
    ax3.grid()
    ax3.legend()
    plt.show()
        
    fig6, ax6 = plt.subplots()
    ax6.plot(range2,loss_v2,linewidth=2.0, label='Finite population model')
    ax6.plot(range2,loss_v2_noF,linewidth=2.0, label='No finite population model')
    #ax.set(xlim=(0, max(vehicles_range)), ylim=(0, 1.02))
    ax6.set_title(f"Loss probability in the queue for zone capacity= {zone_capacity}")
    ax6.set_xlabel("Fleet size")
    ax6.set_ylabel("Loss probability in the queue")
    ax6.grid()
    ax6.legend()
    plt.show()