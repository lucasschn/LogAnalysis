import numpy as np
import pandas as pd
from scipy.interpolate import interp1d as interp1d
import matplotlib.pyplot as plt
import datetime


class segment:

    def __init__(self,mode=None,path2csv=None):
        self.time = np.array([])
        self.cell_voltage = np.array([])
        self.current = np.array([])
        if mode == 'discharge' or mode == 'charge':
            self.mode = mode
        elif mode !=None: 
            raise AttributeError('Argument ''mode'' has to be charge or discharge.')

        if path2csv!=None:
            self.fromcsv(path2csv)

    def concatenate(self,seg2):
        for k in range(len(seg2.stime)):
            seg2.stime[k] += self.stime[-1] + 1.0
        seg = segment()
        seg.stime = self.stime + seg2.stime
        seg.current = self.current.append(seg2.current)
        seg.cell_voltage = self.cell_voltage.append(seg2.cell_voltage)
        if self.mode == seg2.mode:
            seg.mode = self.mode
        else:
            raise Warning('The segments do not have the same mode.')
        return seg

    def count_charge(self,z0):
        dt = np.diff(self.stime)
        self.dq = self.current[1:]*dt
        self.Q = np.sum(self.dq) 
        pdseries_dict = {0:z0}
        cumsum = np.cumsum(self.dq.values)
        if self.mode == 'discharge':
            for k in range(len(self.dq)):
                pdseries_dict.update({k+1:z0 - cumsum[k]/self.Q})
        elif self.mode == 'charge':
            for k in range(len(self.dq)):
                pdseries_dict.update({k+1:z0 + cumsum[k]/self.Q})
        self.z = pd.Series(pdseries_dict)

    def makeseconds(self):
        timelist = []
        stime = []
        for k in range(len(self.time)):
            if len(self.time) < 3600 :
                timelist.append(datetime.datetime.strptime(self.time[k],'             %M:%S,000'))  
            elif len(self.time) < 24*3600 : 
                timelist.append(datetime.datetime.strptime(self.time[k],'             %H:%M:%S,000'))
            else :
                timelist.append(datetime.datetime.strptime(self.time[k],'             %d %H:%M:%S,000'))
            timedelta = timelist[k] - timelist[0]
            stime.append(float(timedelta.total_seconds()))
        self.stime = stime

    def fromcsv(self,csv_file, sep=';', decimal=','):
        self.df = pd.read_csv(csv_file, sep=sep, decimal=decimal, header=1,engine='python')
        self.time = self.df['Time [hh:mm:ss.SSS]']
        self.cell_voltage = self.df['CellVoltage 1 [V]']
        self.current = self.df['Current [A]']
        self.makeseconds()

    def cplot(self,label=None):
        if label!=None:
            plt.plot(self.stime,self.current,label=label)
        else:
            plt.plot(self.stime,self.current)
        plt.grid()

    def vplot(self,label=None):
        if label!=None:
            plt.plot(self.stime,self.cell_voltage,label=label)
        else:
            plt.plot(self.stime,self.cell_voltage)
        plt.grid()

    def zplot(self,label=None):
        if label!=None:
            plt.plot(self.stime,self.z,label=label)
        else:
            plt.plot(self.stime,self.z)
        plt.grid()

class OCVcurve: 

    def __init__(self,path2csv=None):
        self.nb_segments = 0

        if path2csv != None :
            self.setOCVfromcsv(path2csv)    


    def setOCVfromcsv(self,path):
        self.csv_file = pd.read_csv(path)
        self.SOC = self.csv_file['SOC']
        self.OCV = self.csv_file['OCV']
        self.interpOCV = interp1d(self.SOC,self.OCV)
        self.interpSOC = interp1d(self.OCV,self.SOC)

    def fromsegment(self,segment):
        self.SOC = segment.z
        self.OCV = segment.cell_voltage
        self.current = segment.current
        self.mode = segment.mode

    def intrescorr(self,R0):
        ''' Corrects OCV curve with internal resistance R0.'''
        if self.mode == 'charge':
            self.OCV = self.OCV - self.current*R0
        elif self.mode == 'discharge':
            self.OCV = self.OCV + self.current*R0
        

    def getslope(self,z):
        slope = (self.OCVfromSOC(min(1.0,z+0.01))-self.OCVfromSOC(max(0.0,z-0.01)))/(min(1.0,z+0.01)-max(0.0,z-0.01))
        # sorted_SOC = self.SOC.sort_values()
        # slopes = np.diff(self.OCV)/np.diff(self.SOC)
        # for k in range(len(self.SOC)):
        #     if z > sorted_SOC.iloc[-1] or z < sorted_SOC.iloc[0]:
        #         raise ValueError(f'z={z} is out of range ({sorted_SOC.iloc[0]} - {sorted_SOC.iloc[-1]})')
        #     elif z == sorted_SOC.iloc[-1]:
        #         slope = slopes[-1]
        #         break
        #     elif z < sorted_SOC.iloc[k+1]:
        #         slope = slopes[k]
        #         break
        return slope
 

    def OCVfromSOC(self,z):
        v = self.interpOCV(z)
        return v
    

    def SOCfromOCV(self,v):
        z = self.interpSOC(v)
        return z

    def plot(self,color='b',linestyle='-'):
        plt.plot(self.SOC,self.OCV,color=color,linestyle=linestyle)
        plt.xlabel('State of Charge (-)')
        plt.ylabel('Open Circuit Voltage (V)')
        plt.grid()

class Battery:

    def __init__(self,z,Q,eta=1):
        self.total_capacity = Q
        self.charge_state = z
        self.coulombic_efficiency = eta


    def setOCV(OCVcurve):
        self.OCVcurve = OCVcurve

class Thevenin(Battery):

    '''        ____ 
           |--|____|--|   ____
        |--|    R1    |--|____|--
        |  |----||----|    R0    ^
        /\      C1
        \/ OCV                   v(t)
        |
        |________________________v  '''



    def __init__(self,z,Q,OCVcurve,R0,R1,C1,eta=1):
        super().__init__(z,Q,eta=1)
        self.z = z
        self.R0 = R0
        self.R1 = R1 
        self.C1 = C1
        self.OCVcurve = OCVcurve

    def statespace(self,dt): 
        ''' Defines the state-space model matrices for the battery. States are x1=z and x2=iR1'''  
        self.A = np.array([[1., 0.],[0.,  np.exp(-dt/(self.R1*self.C1))]], dtype=float)
        self.B = np.array([[-self.coulombic_efficiency*dt/self.total_capacity],[1 - np.exp(-dt/(self.R1*self.C1))]],dtype=float)
        self.C = np.array([self.OCVcurve.getslope(self.z), -self.R1],dtype=float) # addition of v_empty is missing
        self.D = np.array([-self.R0],dtype=float)

    def reset(self,R0,R1,C1,z):
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1   
        self.z = z     

    def simulate(self,time,current,OCVcurve,plot=False):
        self.simt = time
        self.simi = current
        self.simdt = np.mean(np.diff(self.simt))

        self.simz = []
        self.simz.append(self.z) 
        self.simi1 = [] 
        self.simi1.append(self.simi[0])

        self.simv = []             
        for k in range(len(self.simt)):
            self.simz.append(self.simz[k] - self.coulombic_efficiency*self.simdt/self.total_capacity * self.simi[k])
            self.simi1.append(np.exp(-self.simdt/(self.R1*self.C1)) * self.simi1[k] + (1 - np.exp(-self.simdt/(self.R1*self.C1)))*self.simi[k])
            self.simv.append(OCVcurve.OCVfromSOC(self.simz[k]) - self.R1*self.simi1[k] - self.R0*self.simi[k])

        if plot :
            plt.figure()
            plt.subplot(211)
            plt.plot(time,self.simv,label='simulation')
            plt.legend()
            plt.grid()
            plt.ylabel('Cell voltage (V)')
            plt.subplot(212) 
            plt.plot(time,current)
            plt.xlabel('time (s)')
            plt.ylabel('Current (A)')
            plt.grid()

        return self.simv