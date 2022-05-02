from cProfile import label
from cmath import nan
from city_scenario.city_scenario import Scenario
from network_modelling.network_modelling import Network
from plot_results import Results
import json
import numpy as np
import matplotlib.pyplot as plt

def case_name(scenario_config, charging_policy):
    data_time_interval=np.arange(scenario_config['start_hour'],scenario_config['end_hour'])
    if scenario_config["week_day"]:
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
    case=f'{case_time}_{charging_policy}_{scenario_config["n_charging_stations"]}CS'
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

    #scenario.change_scenario_param(["av_trip_time"], [15])
    network_model=Network(scenario, n_vehicles)
    #network_model.print_stat()
    case=case_name(scenario_config, "opportunistic")
    result_case=Results(scenario, network_model, case)
    result_case.plot_data_on_grid()
    result_case.plot_results([300,500,700])
    result_case.plot_bar_per_zone()
    """
    scenario.change_scenario_param(["av_trip_time"], [60])
    network_model=Network(scenario, n_vehicles)
    #network_model.print_stat()
    case=case_name(scenario_config, "opportunistic")
    result_case=Results(scenario, network_model, case)
    result_case.plot_data_on_grid()
    result_case.plot_results([300,500,700])
    result_case.plot_bar_per_zone()
    """
    plt.show()
    charging_policies= ['opportunistic','closest_CS','uniform_reloc']
    #for ch_policy in charging_policies:
        #network_model=Network(scenario, ch_policy, n_vehicles)
        #case=case_name(scenario_config, charging_policy)
        #result_case=Results(scenario, network_model, case)
        #result_case.plot_data_on_grid()
    #plt.show()
    """
    CS_list=[1,2,4,5,10,20]
    thr=np.zeros((len(CS_list),len(charging_policies)))
    u_d=np.zeros((len(CS_list),len(charging_policies)))
    for i in range(len(CS_list)):
        scenario.change_scenario_param(["n_charging_stations","outlet_per_stations"], [CS_list[i],int(20/CS_list[i])])
        for j in range(len(charging_policies)):
            network_model=Network(scenario, charging_policies[j], n_vehicles)
            thr[i,j]=network_model.ov_throughput
            try:
                u_d[i,j]=network_model.av_ud
            except:
                u_d[i,j]=None
    
    fig1, ax1=plt.subplots()
    fig1.set_size_inches(18.5, 10.5)
    ax_ticks=[str(i)+'/'+str(int(20/i)) for i in CS_list]
    ax1.plot(thr[:,0], label='opportunistic')
    ax1.plot(thr[:,1], label='closest')
    ax1.plot(thr[:,2], label='uniform')
    ax1.set_title("Overal throughput with different concentration of charging stations")
    ax1.set_xlabel("Number of charging stations/Outlet per station")
    ax1.set_ylabel("System throughput")
    ax1.set_xticks(list(np.arange(len(CS_list))))
    ax1.set_xticklabels(ax_ticks)
    ax1.grid()
    ax1.legend(title='Charging policy')
    #plt.savefig(f'img/thr_CS_concentration.png', transparent=True, bbox_inches='tight')

    fig1, ax1=plt.subplots()
    fig1.set_size_inches(18.5, 10.5)
    ax_ticks=[str(i)+'/'+str(int(20/i)) for i in CS_list]
    ax1.plot(u_d[:,0], label='opportunistic')
    ax1.plot(u_d[:,1], label='closest')
    ax1.plot(u_d[:,2], label='uniform')
    ax1.set_title("Average unsatisfied demand with different concentration of charging stations")
    ax1.set_xlabel("Number of charging stations/Outlet per station")
    ax1.set_ylabel("Average unsatisfied demand [%]")
    ax1.set_xticks(list(np.arange(len(CS_list))))
    ax1.set_xticklabels(ax_ticks)
    ax1.grid()
    ax1.legend(title='Charging policy')
    #plt.savefig(f'img/av_ud_CS_concentration.png', transparent=True, bbox_inches='tight')
    plt.show()
    """