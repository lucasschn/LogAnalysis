import pyulog
import numpy as np
from numpy.fft import rfft as rfft, rfftfreq as rfftfreq
import pandas as pd
from pyulgresample.ulogdataframe import DfUlg, TopicMsgs
import datetime

class LogError(Exception):
    def __str__(self):
        return repr(self.args)

def ampspectrum(x):
    """ Returns the magnitudes of the fft for a real x"""
    complex_spectrum = rfft(x)
    X = np.abs(complex_spectrum)
    return X

def avghover(path):
    """ Compatible with pyulgresample"""

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

            average_hovering_rpm = np.mean([average_hovering_rpm1,average_hovering_rpm2,average_hovering_rpm3,average_hovering_rpm4,average_hovering_rpm5,average_hovering_rpm6])

            average_hovering = {'rpm': average_hovering_rpm, 'zthrust': average_hovering_zthrust}

        return average_hovering


def ishover(navstate,stick_in_x,stick_in_y,stick_in_z):
    """ Takes navstate and stick inputs at a certain timestep as arguments. Returns True is the drone is hovering at this moment, False elsewise"""
    ishover = False
    if navstate == 4: # Auto-loiter
        ishover = True 
    elif navstate == 2 and np.all(np.abs([stick_in_x, stick_in_y, stick_in_z]) <= [0.1,0.1,0.1]): # Position control
        ishover = True
    return ishover

default_topic_list = []
default_topic_list.append('sensor_combined')
default_topic_list.append('actuator_outputs')
default_topic_list.append('vehicle_local_position_setpoint')
default_topic_list.append('vehicle_status')
default_topic_list.append('vehicle_land_detected')
default_topic_list.append('manual_control_setpoint')

def logextract(path,topic_list=default_topic_list):
    """ Returns an info dictionnary containing the relevant information in the log "path" according to "topic_list" """
    ulog = pyulog.ULog(path,topic_list) 
    datalist = ulog.data_list # is a list of Data objects, which contain the final topic data for a single topic and instance
    info = {}
    for topic in datalist: 
        if topic.name == 'sensor_combined': 
            data_sc = topic.data
            time_sc = data_sc['timestamp']/1e6 # convert it from us to s
            acc_x=data_sc['accelerometer_m_s2[0]']
            acc_y=data_sc['accelerometer_m_s2[1]']
            acc_z=data_sc['accelerometer_m_s2[2]']
            roll = data_sc['gyro_rad[0]']
            pitch = data_sc['gyro_rad[1]']
            yaw = data_sc['gyro_rad[2]']
            info.update({'time_sc':time_sc, 'acc_x':acc_x, 'acc_y':acc_y,'acc_z':acc_z, 'roll': roll, 'pitch': pitch, 'yaw': yaw})
            info.update (sixaxes_spectrum(info))
        elif topic.name == 'actuator_outputs':
                if (np.all(topic.data['noutputs'] <= 1)): 
                    continue
                else:                     
                    data_ao = topic.data
                    time_ao = data_ao['timestamp']/1e6
                    columns = ['motor 1','motor 2','motor 3','motor 4','motor 5','motor 6']
                    rpm=[]
                    for k in range(np.unique(data_ao['noutputs']).item()):
                        rpm.insert(k,data_ao[f'output[{k}]'])
                    rpm_df = pd.DataFrame(np.array([rpm[0],rpm[1],rpm[2],rpm[3],rpm[4],rpm[5]]).transpose(),index=range(len(time_ao)),columns=columns)
                    info.update({'time_ao':time_ao,'rpm': rpm,'rpm_df': rpm_df})
        elif topic.name == 'vehicle_local_position_setpoint':
            data_vlps = topic.data
            time_vlps = data_vlps['timestamp']/1e6
            vert_thrust = data_vlps['thrust[2]']
            info.update({'time_vlps': time_vlps, 'vert_thrust': vert_thrust})
        elif topic.name == 'vehicle_status':
            data_vs = topic.data
            time_vs = data_vs['timestamp']/1e6
            navstate = data_vs['nav_state']
            info.update({'time_vs': time_vs,'navstate': navstate})
        elif topic.name == 'vehicle_land_detected':
            data_vld = topic.data
        elif topic.name == 'manual_control_setpoint':
            data_mcs = topic.data
            time_mcs = data_mcs['timestamp']/1e6
            stick_in_x = data_mcs['x']
            stick_in_y = data_mcs['y']
            stick_in_z = data_mcs['z']
            info.update({'time_mcs':time_mcs,'stick_in_x': stick_in_x, 'stick_in_y': stick_in_y, 'stick_in_z': stick_in_z})
        elif topic.name == 'battery_status':
            data_bs = topic.data
            time_bs = data_bs['timestamp']/1e6
            battery_current = data_bs['current_a']
            battery_voltage = data_bs['voltage_v']
            battery_filtered_current = data_bs['current_filtered_a']
            battery_filtered_voltage = data_bs['voltage_filtered_v']
            discharged_mah = data_bs['discharged_mah']
            remaining = data_bs['remaining']
            n_cells = data_bs['cell_count']    
            info.update({'time_bs':time_bs,'n_cells':n_cells, 'battery_current':battery_current, 'battery_filtered_current': battery_filtered_current,'battery_voltage': battery_voltage, 'battery_filtered_voltage':battery_filtered_voltage, 'discharged_mah': discharged_mah,'remaining': remaining})
    return info

def logscore(info):
    """ Returns the acceleration score, the peak score and the high-frequencies score computed for the parameters passed as arguments."""

    time = info['time_sc'] # in s
    acc_x = info['acc_x']
    acc_y = info['acc_y'] 
    acc_z = info['acc_z']
    roll = info['roll']
    pitch = info['pitch'] 
    yaw = info['yaw']
    N = len(acc_z)
    spectrum = sixaxes_spectrum(info)


    if (time[-1]-time[0]) < 60:
        raise LogError(f'Time series are too short (less than 1 minute). Consider discarding.')
    else:
        # score calculation for raw acceleration 
        # we want to count the blank area between the zacc and the xacc or yacc as positive points, if they overlap as negative
        max_acc_score = 9.81*len(time) # having constantly the z at gravity and x,y at 0
        acc_score = sum(np.min([acc_x[index],acc_y[index]])-acc_z[index] for index in range(len(time)))/max_acc_score
        print(f'acc score : {acc_score}') # should be above 0.5

        peak_limit = 20 #Hz
        hf_limit = 500 #amplitude

        # score calculation for angles spectrum
        max_peak_score = peak_limit*np.sum((info['P']>=hf_limit) | (info['R']>=hf_limit) | (info['Y']>=hf_limit))
        if max_peak_score == 0:
            peak_score=1
        else:
            peak_score = np.sum([peak_limit - info['freq'][np.argmax([info['P'][index],info['R'][index],info['Y'][index]])] for index in range(N//2) if ((info['P'][index]>=hf_limit) |(info['R'][index]>=hf_limit) |(info['Y'][index]>=hf_limit))])/max_peak_score
        print(f'peak score :{peak_score}')

        max_hf_score = hf_limit*(N - np.argmax(info['freq'].__gt__(peak_limit)))
        # for each frequency above peak_limit, take the distance between hf_limit and the highest curve and sum them
        hf_score = np.sum([hf_limit - np.max([info['R'][index],info['P'][index],info['Y'][index]]) for index in range(N//2) if (info['freq'][index]>peak_limit) ])/max_hf_score
        print(f'hf score : {hf_score}')

        scores = {'acc_score': acc_score,'peak_score': peak_score,'hf_score': hf_score}
        return scores

def sixaxes_spectrum(info):
    """ Takes an info dictionnary and returns the fft for the six axes """ 
    # computing the frequency range of the accelerations
    N = len(info['acc_z']) # number of data points
    dt = np.mean(np.diff(info['time_sc'])) # average sampling time in
    freq = rfftfreq(N,dt) # Hz

    # computing the amplitudes of the accelerations
    Ax = ampspectrum(info['acc_x'])
    Ay = ampspectrum(info['acc_y'])
    Az = ampspectrum(info['acc_z'])

    # computing the amplitudes of the angles
    R = ampspectrum(info['roll'])
    P = ampspectrum(info['pitch'])
    Y = ampspectrum(info['yaw'])

    spectrum = {'freq': freq, 'Ax': Ax, 'Ay': Ay,  'Az': Az, 'R': R, 'P': P, 'Y': Y}
    
    return spectrum