from cProfile import label
from cmath import nan
from city_scenario.city_scenario import Scenario
from network_modelling.network_modelling import Network
from plot_results import Results
import json
import numpy as np
import matplotlib.pyplot as plt

def case_name(scenario_config_file):
    data_time_interval=np.arange(scenario_config_file['start_hour'],scenario_config_file['end_hour'])
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
    
    np.save('thr_CS_concentration_ext.npy', thr)
    np.save('uD_CS_concentration_ext.npy', u_d)
    
    thr = np.load('thr_CS_concentration_ext.npy')
    u_d = np.load('uD_CS_concentration_ext.npy')
    CS_list=[1,2,3,5,6,10,15,30]
    fig1, (ax11,ax21)=plt.subplots(2,1, sharex=True)
    fig1.set_size_inches(18.5, 10.5)
    ax_ticks=[str(i)+'/'+str(int(30/i)) for i in CS_list]
    ax11.plot(thr[:,0], label='opportunistic')
    ax11.plot(thr[:,1], label='closest')
    ax11.plot(thr[:,2], label='uniform')
    fig1.suptitle("Throughput and average unsatisfied demand varying charging stations concentration\nwith different charging policies")
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
    reloc_after_charging=['none','highest_demand','uniform']
    thr=np.zeros((len(charging_policies),len(reloc_after_charging)))
    u_d=np.zeros((len(charging_policies),len(reloc_after_charging)))
    for ch in range(len(charging_policies)):
        for rac in range(len(reloc_after_charging)):
            scenario.change_scenario_param(["charging_policy","reloc_after_charging"],[charging_policies[ch],reloc_after_charging[rac]])
            network_model=Network(scenario,n_vehicles)
            thr[ch,rac]=network_model.ov_throughput
            u_d[ch,rac]=network_model.av_ud

    scenario.change_scenario_param(["order_CS_position_by"],["departure_per_zone"])
    charging_policies= ['opportunistic','closest_CS','uniform']
    reloc_after_charging=['none','highest_demand','uniform']
    thr_2=np.zeros((len(charging_policies),len(reloc_after_charging)))
    u_d_2=np.zeros((len(charging_policies),len(reloc_after_charging)))
    for ch in range(len(charging_policies)):
        for rac in range(len(reloc_after_charging)):
            scenario.change_scenario_param(["charging_policy","reloc_after_charging"],[charging_policies[ch],reloc_after_charging[rac]])
            network_model=Network(scenario,n_vehicles)
            thr_2[ch,rac]=network_model.ov_throughput
            u_d_2[ch,rac]=network_model.av_ud
    
    np.save('thr.npy', thr)
    np.save('u_d.npy', u_d)
    np.save('thr_2.npy', thr_2)
    np.save('u_d_2.npy', u_d_2)
    """
    """
    thr = np.load('thr.npy')
    u_d = np.load('u_d.npy')
    thr_2 = np.load('thr_2.npy')
    u_d_2 = np.load('u_d_2.npy')
    charging_policies= ['opportunistic','closest_CS','uniform']
    reloc_after_charging=['none','highest_demand','uniform']
            
    fig1, ((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2)
    fig1.set_size_inches(18.5, 10.5)
    im11 = ax11.imshow(thr)
    im12 = ax12.imshow(u_d)
    im21 = ax21.imshow(thr_2)
    im22 = ax22.imshow(u_d_2)

    # Show all ticks and label them with the respective list entries
    ax11.set_xticks(np.arange(len(charging_policies)))
    ax11.set_xticklabels(charging_policies)
    ax11.set_yticks(np.arange(len(reloc_after_charging)))
    ax11.set_yticklabels(reloc_after_charging)
    ax12.set_xticks(np.arange(len(charging_policies)))
    ax12.set_xticklabels(charging_policies)
    ax12.set_yticks(np.arange(len(reloc_after_charging)))
    ax12.set_yticklabels(reloc_after_charging)
    ax21.set_xticks(np.arange(len(charging_policies)))
    ax21.set_xticklabels(charging_policies)
    ax21.set_yticks(np.arange(len(reloc_after_charging)))
    ax21.set_yticklabels(reloc_after_charging)
    ax22.set_xticks(np.arange(len(charging_policies)))
    ax22.set_xticklabels(charging_policies)
    ax22.set_yticks(np.arange(len(reloc_after_charging)))
    ax22.set_yticklabels(reloc_after_charging)
    
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax11.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    #plt.setp(ax12.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    #plt.setp(ax21.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    #plt.setp(ax22.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    #ax11.set_title("Random CS placement\nSystem throughput")
    #ax12.set_title("Random CS placement\nAverage unsatisfied demand[%]")
    #ax21.set_title("Max departure per zone CS placement\nSystem throughput")
    #ax22.set_title("Max departure per zone CS placement\nAverage unsatisfied demand[%]")
    rows=['Relocation after charging\n', 'Relocation after charging\n']
    for ax, row in zip((ax11,ax21), rows):
        ax.set_ylabel(row, rotation=90)
    cols=["\nCharging policy","\nCharging policy"]
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

    fig1.suptitle("System throughput and average unsatisfied demand with different charging policies and stations placement")
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
    
    scenario.change_scenario_param(["delay_queue"],[False])
    net=Network(scenario)
    #net.print_stat()
    case="!noD_"+case_name(scenario_config)
    res=Results(scenario,net,case)
    res.plot_data_on_grid()
    res.plot_bar_per_zone()
    res.plot_results([300,600,900,1200])

    scenario.change_scenario_param(["delay_queue"],["multiple"])
    net=Network(scenario)
    #net.print_stat()
    case="!mD_"+case_name(scenario_config)
    res=Results(scenario,net,case)
    res.plot_data_on_grid()
    res.plot_bar_per_zone()
    res.plot_results([300,600,900,1200])

    scenario.change_scenario_param(["delay_queue"],["single"])
    net2=Network(scenario)
    #net.print_stat()
    case="!sD_"+case_name(scenario_config)
    res=Results(scenario,net2,case)
    res.plot_data_on_grid()
    res.plot_bar_per_zone()
    res.plot_results([300,600,900,1200])

    #plt.show()
    #net.fluxes_with_reloc4charg_del()
    
   