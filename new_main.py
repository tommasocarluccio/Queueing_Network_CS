
from turtle import color
from city_scenario.city_scenario import Scenario
from network_modelling.network_modelling import Network
from plot_results import Results
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.ticker as ticker
import pandas as pd

def case_name(scenario_config_file):
    data_time_interval=np.arange(scenario_config_file['start_hour'],scenario_config_file['end_hour'])
    if scenario_config_file["week_day"]:
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
    case=f'{case_time}_{scenario_config_file["charging_policy"]}_{scenario_config_file["reloc_after_charging"]}_{scenario_config_file["n_charging_stations"]}CS'
    return case


if __name__=='__main__':
    np.random.seed(42)
    plt.rcParams.update({'figure.max_open_warning': 0})

    scenario_config=json.load(open("city_scenario/scenario_config.json","r"))
    scenario=Scenario("city_scenario/scenario_config.json")
    #scenario.print_scenario()
    #print(scenario.zones_with_charging_stations)
    #scenario.change_scenario_param("n_charging_stations", 10)
    #print(scenario.zones_with_charging_stations)
    #scenario.print_scenario()

    #charging policies: ['opportunistic','closest_CS','uniform_reloc']
    #charging_policy='opportunistic'
    n_vehicles=400
    #network_model=Network(scenario, charging_policy, n_vehicles)
    #network_model.print_stat()

    #case=case_name(scenario_config, charging_policy)
    #result_case=Results(scenario, network_model, case)
    #result_case.plot_data_on_grid()
    #result_case.plot_results([300,500,700])
    #result_case.plot_bar_per_zone()
    #plt.show()
    """ 
    #scenario.change_scenario_param(["av_trip_time"], [15])
    network_model=Network(scenario, n_vehicles)
    #network_model.print_stat()
    case=case_name(scenario_config, "opportunistic")
    result_case=Results(scenario, network_model, case)
    result_case.plot_data_on_grid()
    result_case.plot_results([300,500,700])
    result_case.plot_bar_per_zone()
    
    scenario.change_scenario_param(["av_trip_time"], [60])
    network_model=Network(scenario, n_vehicles)
    #network_model.print_stat()
    case=case_name(scenario_config, "opportunistic")
    result_case=Results(scenario, network_model, case)
    result_case.plot_data_on_grid()
    result_case.plot_results([300,500,700])
    result_case.plot_bar_per_zone()
    
    plt.show()
    charging_policies= ['opportunistic','closest_CS','uniform_reloc']
    #for ch_policy in charging_policies:
        #network_model=Network(scenario, ch_policy, n_vehicles)
        #case=case_name(scenario_config, charging_policy)
        #result_case=Results(scenario, network_model, case)
        #result_case.plot_data_on_grid()
    #plt.show()
    """
    #CS concentration
    """
    charging_policies= ['opportunistic','closest_CS','uniform']
    CS_list=[1,2,3,5,6,10,15,30]
    thr=np.zeros((len(CS_list),len(charging_policies)))
    u_d=np.zeros((len(CS_list),len(charging_policies)))
    for i in range(len(CS_list)):
        scenario.change_scenario_param(["n_charging_stations","outlet_per_stations"], [CS_list[i],int(30/CS_list[i])])
        for j in range(len(charging_policies)):
            scenario.change_scenario_param(['charging_policy'],[charging_policies[j]])
            network_model=Network(scenario)
            thr[i,j]=network_model.ov_throughput
            try:
                u_d[i,j]=network_model.av_ud
            except:
                u_d[i,j]=None
    
    np.save('thr_CS_concentration_12_prob.npy', thr)
    np.save('uD_CS_concentration_12_prob.npy', u_d)
    
    thr = np.load('thr_CS_concentration_12_prob.npy')
    u_d = np.load('uD_CS_concentration_12_prob.npy')

    CS_list=[1,2,3,5,6,10,15,30]
    fig1, (ax11,ax21)=plt.subplots(2,1, sharex=True)
    fig1.set_size_inches(18.5, 10.5)
    ax_ticks=[str(i)+'/'+str(int(30/i)) for i in CS_list]
    ax11.plot(thr[:,0], label='opportunistic')
    ax11.plot(thr[:,1], label='closest')
    ax11.plot(thr[:,2], label='uniform')
    fig1.suptitle("Throughput and average unsatisfied demand varying charging stations concentration\nwith different charging policies and probabilistic relocation after charging")
    ax11.set_xlabel("Number of charging stations/Outlet per station")
    ax11.set_ylabel("System throughput")
    ax11.set_xticks(list(np.arange(len(CS_list))))
    ax11.set_xticklabels(ax_ticks)
    ax11.grid()
    ax11.legend(title='Charging policy')
    ax21.plot(u_d[:,0], label='opportunistic')
    ax21.plot(u_d[:,1], label='closest')
    ax21.plot(u_d[:,2], label='uniform')
    #ax21.set_title("Average unsatisfied demand with different concentration of charging stations")
    ax21.set_xlabel("Number of charging stations/Outlet per station")
    ax21.set_ylabel("Average unsatisfied demand [%]")
    ax21.set_xticks(list(np.arange(len(CS_list))))
    ax21.set_xticklabels(ax_ticks)
    ax21.grid()
    ax21.legend(title='Charging policy')
    #plt.savefig(f'img/thr_CS_concentration.png', transparent=True, bbox_inches='tight')

    
    #plt.savefig(f'img/av_ud_CS_concentration.png', transparent=True, bbox_inches='tight')
    plt.show()
    """
    #heatmaps
    """
    scenario.change_scenario_param(["order_CS_position_by"],["random"])
    charging_policies= ['opportunistic','closest_CS','uniform']
    reloc_after_charging=['none','highest_demand','uniform','probabilistic']
    thr=np.zeros((len(charging_policies),len(reloc_after_charging)))
    u_d=np.zeros((len(charging_policies),len(reloc_after_charging)))
    #for t in range(0,24):
    for ch in range(len(charging_policies)):
        for rac in range(len(reloc_after_charging)):
            #scenario.change_scenario_param(["start_hour","end_hour","charging_policy","reloc_after_charging"],[t,t+1,charging_policies[ch],reloc_after_charging[rac]])
            scenario.change_scenario_param(["charging_policy","reloc_after_charging"],[charging_policies[ch],reloc_after_charging[rac]])
            network_model=Network(scenario)
            thr[ch,rac]+=network_model.ov_throughput
            u_d[ch,rac]+=network_model.av_ud
    #thr=thr/24
    #u_d=u_d/24

    np.save('thr_day.npy', thr)
    np.save('ud_day.npy', u_d)
    
    scenario.change_scenario_param(["order_CS_position_by"],["departure_per_zone"])
    charging_policies= ['opportunistic','closest_CS','uniform']
    reloc_after_charging=['none','highest_demand','uniform','probabilistic']
    thr_2=np.zeros((len(charging_policies),len(reloc_after_charging)))
    u_d_2=np.zeros((len(charging_policies),len(reloc_after_charging)))
    #for t in range(0,24):
    for ch in range(len(charging_policies)):
        for rac in range(len(reloc_after_charging)):
            #scenario.change_scenario_param(["start_hour","end_hour","charging_policy","reloc_after_charging"],[t,t+1,charging_policies[ch],reloc_after_charging[rac]])
            scenario.change_scenario_param(["charging_policy","reloc_after_charging"],[charging_policies[ch],reloc_after_charging[rac]])
            network_model=Network(scenario)
            thr_2[ch,rac]+=network_model.ov_throughput
            u_d_2[ch,rac]+=network_model.av_ud
    #thr_2=thr_2/24
    #u_d_2=u_d_2/24
    
    np.save('thr_day_rdn.npy', thr_2)
    np.save('ud_day_rdn.npy', u_d_2)
    """
    #count over 24 matrices
    """
    complete_thr_random=np.load('thr_complete_random.npy')
    complete_ud_random=np.load('u_d_complete_random.npy')
    complete_thr=np.load('thr_2_day.npy')
    complete_ud=np.load('u_d_2_day.npy')
    #print(complete_thr)
    dict_map={}
    id=0
    charging_policies= ['opportunistic','closest_CS','uniform']
    reloc_after_charging=['none','highest_demand','uniform','probabilistic']
    for ch in range(len(charging_policies)):
            for rac in range(len(reloc_after_charging)):
                dict_map[id]=charging_policies[ch]+'+'+reloc_after_charging[rac]
                id+=1    
    best_thr_idx=[]
    best_thr_idx_rnd=[]
    top_num=1
    for t in range (24):
        best_thr_idx.extend(list(np.argpartition(complete_thr[t].flatten(),-top_num)[-top_num:]))
        best_thr_idx_rnd.extend(list(np.argpartition(complete_thr_random[t].flatten(),-top_num)[-top_num:]))
    best_thr_idx=np.array([dict_map[i] for i in best_thr_idx])
    best_thr_idx_rnd=np.array([dict_map[i] for i in best_thr_idx_rnd])
    print("\n")
    print(Counter(best_thr_idx))
    print("\n")
    print(Counter(best_thr_idx_rnd))
    """
    #plot heatmap
    """
    #balanced matrix
    #thr = np.load('thr_day.npy')
    #u_d = np.load('ud_day.npy')
    #thr_2 = np.load('thr_day_rdn.npy')
    #u_d_2 = np.load('ud_day_rdn.npy')
    #averaged hourly matrix
    thr_2=complete_thr.sum(axis=0)/24
    u_d_2=complete_ud.sum(axis=0)/24
    thr=complete_thr_random.sum(axis=0)/24
    u_d=complete_ud_random.sum(axis=0)/24
    #one hour only
    thr_2=complete_thr[12]
    u_d_2=complete_ud[12]
    thr=complete_thr_random[12]
    u_d=complete_ud_random[12]

    charging_policies= ['opportunistic','closest_CS','uniform']
    reloc_after_charging=['none','highest_demand','uniform','probabilistic']
            
    fig1, ((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2)
    fig1.set_size_inches(18.5, 10.5)
    im11 = ax11.imshow(thr)
    im12 = ax12.imshow(u_d)
    im21 = ax21.imshow(thr_2)
    im22 = ax22.imshow(u_d_2)

    # Show all ticks and label them with the respective list entries
    ax11.set_xticks(np.arange(len(reloc_after_charging)))
    ax11.set_xticklabels(reloc_after_charging)
    ax11.set_yticks(np.arange(len(charging_policies)))
    ax11.set_yticklabels(charging_policies)
    ax12.set_xticks(np.arange(len(reloc_after_charging)))
    ax12.set_xticklabels(reloc_after_charging)
    ax12.set_yticks(np.arange(len(charging_policies)))
    ax12.set_yticklabels(charging_policies)
    ax21.set_xticks(np.arange(len(reloc_after_charging)))
    ax21.set_xticklabels(reloc_after_charging)
    ax21.set_yticks(np.arange(len(charging_policies)))
    ax21.set_yticklabels(charging_policies)
    ax22.set_xticks(np.arange(len(reloc_after_charging)))
    ax22.set_xticklabels(reloc_after_charging)
    ax22.set_yticks(np.arange(len(charging_policies)))
    ax22.set_yticklabels(charging_policies)
    
    rows=["Charging policy\n", "Charging policy\n"]
    for ax, row in zip((ax11,ax21), rows):
        ax.set_ylabel(row, rotation=90)
    cols=['\nRelocation after charging','\nRelocation after charging']
    for ax, col in zip((ax21,ax22), cols):
        ax.set_xlabel(col, rotation=0)
    pad = 5
    col_title=["System throughput\n","Average unsatisfied demand\n"]
    for ax, col in zip((ax11,ax12), col_title):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline', fontweight="bold")
    row_title=["Random CS     \nplacement     ","Max departure     \nper zone     \nCS placement     "]    
    for ax, row in zip((ax11,ax21), row_title):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center', fontweight="bold")
    #COLORBAR
    cbar11 = ax11.figure.colorbar(im11, ax=ax11)
    cbar11.ax.set_ylabel("System throughput", rotation=-90, va="bottom")
    cbar12 = ax12.figure.colorbar(im12, ax=ax12)
    cbar12.ax.set_ylabel("Average unsatisfied demand", rotation=-90, va="bottom")
    cbar21 = ax21.figure.colorbar(im21, ax=ax21)
    cbar21.ax.set_ylabel("System throughput", rotation=-90, va="bottom")
    cbar22 = ax22.figure.colorbar(im22, ax=ax22)
    cbar22.ax.set_ylabel("Average unsatisfied demand", rotation=-90, va="bottom")
    # Loop over data dimensions and create text annotations.
    for i in range(len(charging_policies)):
        for j in range(len(reloc_after_charging)):
            text = ax11.text(j, i, np.round(thr[i, j],1),ha="center", va="center", color="w")
            text = ax12.text(j, i, f"{np.round(u_d[i, j],1)}%",ha="center", va="center", color="w")
            text = ax21.text(j, i, np.round(thr_2[i, j],1),ha="center", va="center", color="w")
            text = ax22.text(j, i, f"{np.round(u_d_2[i, j],1)}%",ha="center", va="center", color="w")

    fig1.suptitle("System throughput and average unsatisfied demand with different charging policies and stations placement\nHourly matrix (12-1pm)")
    #fig1.tight_layout()
    fig1.subplots_adjust(left=0.2)
    plt.show()
    """
    #as n_vehicles
    """
    charging_policies=['opportunistic','closest_CS','uniform']
    range_vehicles=list(np.arange(200,1250,50))
    thr=np.zeros((len(charging_policies),len(range_vehicles)))
    u_d=np.zeros((len(charging_policies),len(range_vehicles)))
    for ch in range(len(charging_policies)):
            for nv in range(len(range_vehicles)):
                scenario.change_scenario_param(["charging_policy","n_vehicles"],[charging_policies[ch],int(range_vehicles[nv])])
                network_model=Network(scenario)
                thr[ch,nv]=network_model.ov_throughput
                u_d[ch,nv]=network_model.av_ud
    np.save('thr_v_h12_randomCS.npy', thr)
    np.save('u_d_v_h12_randomCS.npy', u_d)

    #range_vehicles=list(np.arange(200,2100,100))
    range_vehicles=list(np.arange(200,1250,50))
    thr_v = np.load('thr_v_h12_randomCS.npy')
    u_d_v = np.load('u_d_v_h12_randomCS.npy')
    fig2, (ax21,ax22)=plt.subplots(2,1, sharex=True)
    fig2.set_size_inches(18.5, 10.5)
    ax21.plot(thr_v[0,:],label='opportunistic')
    ax21.plot(thr_v[1,:], label='closest CS')
    ax21.plot(thr_v[2,:], label='uniform')
    ax22.plot(u_d_v[0,:],label='opportunistic')
    ax22.plot(u_d_v[1,:], label='closest CS')
    ax22.plot(u_d_v[2,:], label='uniform')
    ax21.set_xticks(list(np.arange(len(range_vehicles))))
    ax21.set_xticklabels(range_vehicles)
    ax22.set_xticks(list(np.arange(len(range_vehicles))))
    ax22.set_xticklabels(range_vehicles)
    ax21.set_xlabel("Number of vehicles")
    ax21.set_ylabel("System throughput")
    ax21.legend(title='Charging policy')
    ax22.set_xlabel("Number of vehicles")
    ax22.set_ylabel("Average unsatisfied demand [%]")
    ax22.legend(title='Charging policy')
    ax21.grid()
    ax22.grid()
    fig2.suptitle("Throughput and average unsatisfied demand varying number of vehicles\nwith different charging policies, no relocation after charging\nand random placement of charging stations\nhourly data (12-1 pm)")
    #fig2.suptitle("Throughput and average unsatisfied demand varying number of vehicles\nwith different charging policies, no relocation after charging\nand random placement of charging stations")
    plt.show()
    """
    #delays
    """
    scenario.change_scenario_param(["delay_queue"],[False])
    net=Network(scenario)
    #net.print_stat()
    case="!noD_"+case_name(scenario_config)
    res=Results(scenario,net,case)
    res.plot_data_on_grid()
    #res.plot_bar_per_zone()
    #res.plot_results([300,600,900,1200])

    scenario.change_scenario_param(["delay_queue"],["multiple"])
    net=Network(scenario)
    #net.print_stat()
    case="!mD_"+case_name(scenario_config)
    res=Results(scenario,net,case)
    res.plot_data_on_grid()
    #res.plot_bar_per_zone()
    #res.plot_results([300,600,900,1200])

    scenario.change_scenario_param(["delay_queue"],["single"])
    net2=Network(scenario)
    #net.print_stat()
    case="!sD_"+case_name(scenario_config)
    res=Results(scenario,net2,case)
    res.plot_data_on_grid()
    #res.plot_bar_per_zone()
    #res.plot_results([300,600,900,1200])

    plt.show()
    #net.fluxes_with_reloc4charg_del()
    """
    """
    scenario.change_scenario_param(['delay_queue'],['multiple'])
    av_trip_time_range=list(np.arange(10,155,5))
    thr_increasing_delay=np.zeros(len(av_trip_time_range))
    av_v_increasing_delay=np.zeros(len(av_trip_time_range))
    #for t in range(0,24):
    for av_t in range(len(av_trip_time_range)):
        #scenario.change_scenario_param(['start_hour','end_hour','av_trip_time'],[t,t+1,int(av_trip_time_range[av_t])])
        scenario.change_scenario_param(['av_trip_time'],[int(av_trip_time_range[av_t])])
        net=Network(scenario)
        thr_increasing_delay[av_t]+=net.ov_throughput
        #print("Thr: ", net.ov_throughput)
        av_v_increasing_delay[av_t]+=(np.sum(net.av_vehicles[net.n_zones+net.n_charging_stations:net.n_zones*2+net.n_charging_stations]))
    #thr_increasing_delay=thr_increasing_delay/24
    #av_v_increasing_delay=av_v_increasing_delay/24
        
    np.save('thr_increasing_delay_12.npy', thr_increasing_delay)
    np.save('av_v_increasing_delay_12.npy', av_v_increasing_delay)
    
    thr_increasing_delay = np.load('thr_increasing_delay_12.npy')
    av_v_increasing_delay=np.load('av_v_increasing_delay_12.npy')
    
    scenario.change_scenario_param(['delay_queue'],['false'])
    thr_no_delay=0
    #for t in range(0,24):
    #scenario.change_scenario_param(['start_hour','end_hour'],[t,t+1])
    net=Network(scenario)
    thr_no_delay+=net.ov_throughput
    #thr_no_delay=thr_no_delay/24

    thr_increasing_delay=np.append(thr_no_delay, thr_increasing_delay)
    av_v_increasing_delay=np.append(0,av_v_increasing_delay)
    av_trip_time_range=[0]+list(np.arange(10,155,5))
    fig1,(ax11,ax21)=plt.subplots(2,1)
    fig1.set_size_inches(18.5, 10.5)
    ax11.plot(thr_increasing_delay, color='red')
    ax21.plot(av_v_increasing_delay, color='green')
    ax11.set_xlabel('Average trip time (min)')
    ax11.set_ylabel('Throughput')
    ax21.set_xlabel('Average trip time (min)')
    ax21.set_ylabel('Vehicles')
    fig1.suptitle('System throughput and vehicles in delay zones with increasing averge trip time\nmean values of 24 hourly matrices')
    ax11.set_xticks(list(np.arange(len(av_trip_time_range))))
    ax11.set_xticklabels(av_trip_time_range)
    ax21.set_xticks(list(np.arange(len(av_trip_time_range))))
    ax21.set_xticklabels(av_trip_time_range)
    ax11.set_title('System throughput')
    ax21.set_title('Average number of vehicles in delay zones')
    ax11.grid()
    ax21.grid()
    plt.show()
    """
    #power grid
    """
    tot_consumption=np.zeros(24)
    scenario.change_scenario_param(["av_consumption"],[.14])
    for t in range(0,24):
        scenario.change_scenario_param(["start_hour","end_hour"],[t,t+1])
        net=Network(scenario)
        print(np.sum(net.city_grid_CS_data['av_vehicles']))
        print(np.sum(net.city_grid_CS_data['av_server_occupancy']))
        #print(np.sum(net.city_grid_CS_data['av_server_occupancy']*20))
        tot_consumption[t]=np.sum(net.city_grid_CS_data['av_server_occupancy']*scenario_config['outlet_power'])
        #print(tot_consumption)
    np.save('hourly_consumption_014.npy', tot_consumption)
    """
    """
    #varying av consumption with av_trip_length=4km
    tot_consumption_014=np.load('hourly_consumption_014.npy')
    tot_consumption_017=np.load('hourly_consumption_017.npy')
    tot_consumption_020=np.load('hourly_consumption_020.npy')

    tot_consumption_3=np.load('hourly_consumption_3.npy')
    tot_consumption_4=np.load('hourly_consumption_4.npy')
    tot_consumption_5=np.load('hourly_consumption_5.npy')

    fig,(ax1,ax2)=plt.subplots(2,1)
    fig.set_size_inches(18.5, 10.5)
    ax1.plot(np.arange(24),tot_consumption_014, color='blue', label='0.14kWh/km')
    ax1.plot(np.arange(24),tot_consumption_017, color='green', label='0.17kWh/km')
    ax1.plot(np.arange(24),tot_consumption_020, color='red', label='0.20kWh/km')
    
    ax2.plot(np.arange(24),tot_consumption_3, color='blue', label='3km')
    ax2.plot(np.arange(24),tot_consumption_4, color='green', label='4km')
    ax2.plot(np.arange(24),tot_consumption_5, color='red', label='5km')
    
    ax1.set_xlabel('Hour of day')
    ax1.set_ylabel('Consumption [kWh]')
    ax1.set_title('Fixed average trip length= 4km')
    ax2.set_xlabel('Hour of day')
    ax2.set_ylabel('Consumption [kWh]')
    ax2.set_title('Fixed average vehicle consumption= 0.17kWh/km')

    ax1.set_xticks(list(np.arange(24)))
    ax1.set_xticklabels(list(np.arange(24)))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.set_xticks(list(np.arange(24)))
    ax2.set_xticklabels(list(np.arange(24)))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    fig.suptitle('Average system power consumption for charging operations during the day')
    ax1.grid()
    ax2.grid()
    ax1.legend(title='Average vehicle consumption')
    ax2.legend(title='Average trip length')
    plt.show()
    """
   
    #global indexes over day
    """
    day_profile=pd.DataFrame(index=np.arange(24),columns=['ov_thr','av_ud','min_thr','max_thr','min_veh','max_veh','avg_veh','min_ut','max_ut'])
    for t in range (24):
        scenario.change_scenario_param(['start_hour','end_hour'],[t,t+1])
        net=Network(scenario)
        day_profile.iloc[t]=[net.ov_throughput,net.av_ud,
        net.city_grid_data['throughput'].min(),net.city_grid_data['throughput'].max(),
        net.city_grid_data['av_vehicles'].min(),net.city_grid_data['av_vehicles'].max(),net.city_grid_data['av_vehicles'].mean(),
        net.city_grid_data['utilization'].min(),net.city_grid_data['utilization'].max()]
    day_profile.to_csv('day_profile.csv')
    """
    """
    day_profile=pd.read_csv('day_profile.csv')
    print(day_profile)
    
    
    fig1,(ax11,ax12,ax13)=plt.subplots(3,1)
    fig1.set_size_inches(18.5, 10.5)
    ax11.bar(np.arange(24),day_profile['ov_thr'], color='green')
    ax12.bar(np.arange(24),day_profile['av_ud'], color='red')
    ax13.bar(np.arange(24),day_profile['max_veh'],color='blue')
    ax11.set_xlabel('Hour of day')
    ax12.set_xlabel('Hour of day')
    ax13.set_xlabel('Hour of day')
    ax11.set_ylabel('Throughput')
    ax12.set_ylabel("Unsatisfied demand [%]")
    ax13.set_ylabel("Number of vehicles")
    ax11.set_title('System throughput per hour of day')
    ax12.set_title("Average unsatisfied mobility demand per hour of day")
    ax13.set_title("Maximum number of vehicles in a single zone per hour of day")
    ax11.set_xticks(list(np.arange(24)))
    ax11.set_xticklabels(list(np.arange(24)))
    ax12.set_xticks(list(np.arange(24)))
    ax12.set_xticklabels(list(np.arange(24)))
    ax13.set_xticks(list(np.arange(24)))
    ax13.set_xticklabels(list(np.arange(24)))
    ax11.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax12.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax13.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax11.grid()
    ax12.grid()
    ax13.grid()
    plt.subplot_tool()
    plt.show()
    """
    #CS dimensioning
    """
    scenario.print_scenario()
    CS_range=np.arange(1,31)
    CS_num_avgs=pd.DataFrame(index=CS_range,columns=['mean_thr','mean_ut','mean_ser_occ','mean_del','mean_veh','mean_wait'])
    for n_stations in CS_range:
        scenario.change_scenario_param(['n_charging_stations'],[int(n_stations)])
        scenario.print_scenario()
        net=Network(scenario)
        CS_num_avgs.loc[n_stations]=[net.city_grid_CS_data['throughput'].mean(),net.city_grid_CS_data['utilization'].mean(),
        net.city_grid_CS_data['av_server_occupancy'].mean(),net.city_grid_CS_data['av_delay'].mean(),
        net.city_grid_CS_data['av_vehicles'].mean(),net.city_grid_CS_data['av_waiting'].mean()]
        
    CS_num_avgs.to_csv('CS_num_avgs_1outlet_12.csv')
    print(CS_num_avgs)
    

    CS_num_avgs=pd.read_csv('CS_num_avgs_1outlet_12.csv')
    CS_range=np.arange(1,31)
    #case='1_12wd'
    #res=Results(scenario,net, case)
    #res.plot_data_on_grid()
    #res.plot_results([400,600,800,1000])
    #res.plot_bar_per_zone()
    #plt.show()
    
    fig1, ax1=plt.subplots()
    fig1.set_size_inches(18.5, 10.5)
    ax1.bar(CS_range,CS_num_avgs['mean_del'], label='Total time', color='blue')
    ax1.bar(CS_range,CS_num_avgs['mean_wait'], label='Waiting time', color='green', width=0.6)
    ax1.set_xlabel('Number of CS')
    ax1.set_ylabel("Time [h]")
    ax1.set_title("Average total time and average waiting time spent in CS")
    ax1.set_xticks(list(CS_range))
    ax1.set_xticklabels(list(CS_range))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.grid()
    ax1.legend()
    plt.show()
    
    fig1,(ax11,ax12,ax13)=plt.subplots(3,1)
    fig1.set_size_inches(18.5, 10.5)
    ax11.bar(CS_range,CS_num_avgs['mean_thr'], color='green')
    ax12.bar(CS_range,CS_num_avgs['mean_ut']*100, color='red')
    ax13.bar(CS_range,CS_num_avgs['mean_ser_occ'],color='blue')
    ax11.set_xlabel('Number of CS')
    ax12.set_xlabel('Number of CS')
    ax13.set_xlabel('Number of CS')
    ax11.set_ylabel('Throughput')
    ax12.set_ylabel("Utilization [%]")
    ax13.set_ylabel("Number of vehicles")
    ax11.set_title('Average CS throughput')
    ax12.set_title("Average CS utilization")
    ax13.set_title("Average number of vehicles in charge per hour")
    ax11.set_xticks(list(CS_range))
    ax11.set_xticklabels(list(CS_range))
    ax12.set_xticks(list(CS_range))
    ax12.set_xticklabels(list(CS_range))
    ax13.set_xticks(list(CS_range))
    ax13.set_xticklabels(list(CS_range))
    ax11.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax12.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax13.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax11.grid()
    ax12.grid()
    ax13.grid()
    plt.subplot_tool()
    plt.show()
    """

    #charging network studies
    scenario.print_scenario()
    net=Network(scenario)
    case="ch_network1_12_uni"
    print(net.city_grid_data['av_vehicles'].max())
    print(len(net.city_grid_data.loc[net.city_grid_data['un_demand']>90].index))
    print(net.city_grid_data['lost_req'].sum())
    res=Results(scenario,net,case)
    res.plot_data_on_grid()
    res.plot_city_zones(True,True)
    res.plot_results([200,400,600,800])
    res.plot_bar_per_zone()
    #plt.show()