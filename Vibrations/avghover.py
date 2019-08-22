from pyulgresample.ulogdataframe import DfUlg, TopicMsgs
from analog import ishover as ishover
import numpy as np

def avghover(path):
    topic_list = []
    topic_list.append('vehicle_status')
    topic_list.append('vehicle_land_detected')
    topic_list.append('manual_control_setpoint')

    dfulg = DfUlg.create(log_file,topic_list)

    msglist = []
    for k in range(6):
        msglist.append(f'F_output_{k}')

    rpm1 = dfulg.df[f'T_actuator_outputs_0__{msglist[0]}']
    rpm2 = dfulg.df[f'T_actuator_outputs_0__{msglist[1]}']
    rpm3 = dfulg.df[f'T_actuator_outputs_0__{msglist[2]}']
    rpm4 = dfulg.df[f'T_actuator_outputs_0__{msglist[3]}']
    rpm5 = dfulg.df[f'T_actuator_outputs_0__{msglist[4]}']
    rpm6 = dfulg.df[f'T_actuator_outputs_0__{msglist[5]}']

    stick_in_x = dfulg.df['T_manual_control_setpoint_0__F_x']
    stick_in_y = dfulg.df['T_manual_control_setpoint_0__F_y']
    stick_in_z = dfulg.df['T_manual_control_setpoint_0__F_z']

    navstate = dfulg.df['T_vehicle_status_0__F_nav_state']


    hovering_bool=[]

    for k in range(len(navstate)):
        hovering_bool.append(ishover(navstate[k],stick_in_x[k],stick_in_y[k],stick_in_z[k]))

    average_hovering_rpm1 = np.mean(rpm1[hovering_bool])
    average_hovering_rpm2 = np.mean(rpm2[hovering_bool])
    average_hovering_rpm3 = np.mean(rpm3[hovering_bool])
    average_hovering_rpm4 = np.mean(rpm4[hovering_bool])
    average_hovering_rpm5 = np.mean(rpm5[hovering_bool])
    average_hovering_rpm6 = np.mean(rpm6[hovering_bool])

    average_hovering_rpm=np.mean([average_hovering_rpm1,average_hovering_rpm2,average_hovering_rpm3,average_hovering_rpm4,average_hovering_rpm5,average_hovering_rpm6])

    return average_hovering_rpm