from pyulgresample.ulogdataframe import DfUlg, TopicMsgs
import sys 
sys.path.append('/home/lucas/Documents/Log_Analysis')

from analog import ishover as ishover, LogError
import numpy as np
import datetime 

def avghover(path):
    topic_list = []
    topic_list.append('vehicle_status')
    topic_list.append('actuator_outputs')
    topic_list.append('manual_control_setpoint')
    topic_list.append('vehicle_local_position')
    topic_list.append('vehicle_local_position_setpoint')

    dfulg = DfUlg.create(path,topic_list)

    msglist = []
    for k in range(6):
        msglist.append(f'F_output_{k}')

    z = dfulg.df['T_vehicle_local_position_0__F_z']

    if (z.index[-1]-z.index[0]) < datetime.timedelta(minutes=1):
        raise LogError(f'Log {path} is too short (less than 1 minute). Consider discarding.')
    else :

        rpm1 = dfulg.df[f'T_actuator_outputs_0__{msglist[0]}']
        rpm2 = dfulg.df[f'T_actuator_outputs_0__{msglist[1]}']
        rpm3 = dfulg.df[f'T_actuator_outputs_0__{msglist[2]}']
        rpm4 = dfulg.df[f'T_actuator_outputs_0__{msglist[3]}']
        rpm5 = dfulg.df[f'T_actuator_outputs_0__{msglist[4]}']
        rpm6 = dfulg.df[f'T_actuator_outputs_0__{msglist[5]}']

        T = dfulg.df[f'T_vehicle_local_position_setpoint_0__F_thrust_2']
        
        stick_in_x = dfulg.df['T_manual_control_setpoint_0__F_x']
        stick_in_y = dfulg.df['T_manual_control_setpoint_0__F_y']
        stick_in_z = dfulg.df['T_manual_control_setpoint_0__F_z']

        navstate = dfulg.df['T_vehicle_status_0__F_nav_state']


        hovering_bool=[]

        for k in range(len(navstate)):
            hovering_bool.append(ishover(navstate[k],stick_in_x[k],stick_in_y[k],stick_in_z[k]))

        alt = -(z[np.isnan(z) == False][0] - z) # altitude relative to the first non-nan local z-position
        max_alt = np.max(alt)

        if not np.any(hovering_bool and max_alt < 1):
            
            raise LogError(f'The drone does not hover in log {path}. \n Consider discarding this file.')
        
        else:
            
            average_hovering_rpm1 = np.mean(rpm1[hovering_bool and alt < 1])
            average_hovering_rpm2 = np.mean(rpm2[hovering_bool and alt < 1])
            average_hovering_rpm3 = np.mean(rpm3[hovering_bool and alt < 1])
            average_hovering_rpm4 = np.mean(rpm4[hovering_bool and alt < 1])
            average_hovering_rpm5 = np.mean(rpm5[hovering_bool and alt < 1])
            average_hovering_rpm6 = np.mean(rpm6[hovering_bool and alt < 1])
            
            average_hovering_zthrust = np.mean(T[hovering_bool and alt < 1])

            average_hovering_rpm=np.mean([average_hovering_rpm1,average_hovering_rpm2,average_hovering_rpm3,average_hovering_rpm4,average_hovering_rpm5,average_hovering_rpm6])

            average_hovering = {'rpm': average_hovering_rpm, 'zthrust': average_hovering_zthrust}

    return average_hovering



# here the test can be conducted for a specific file 
log_path = '/home/lucas/Documents/Log_Analysis/Logs'
log_index = '270'
log_date = '2019-8-19'
log_time = '14_33_43'
test_file = f'{log_path}/log_{log_index}_{log_date}-{log_time}.ulg'
test_file = f'{log_path}/{log_time}.ulg'

try :
    print(avghover(test_file))
except LogError:
    print('file discarded')
