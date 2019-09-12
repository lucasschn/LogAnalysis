import numpy as np
import pandas as pd
from scipy.interpolate import interp1d as interp1d
import matplotlib.pyplot as plt

class OCVcurve: 

    def __init__(self,path2csv=None):
        if path2csv != None :
            self.setOCVfromcsv(path2csv)

    def setOCVfromcsv(self,path):
        self.csv_file = pd.read_csv(path)
        self.SOC = self.csv_file['SOC']
        self.OCV = self.csv_file['OCV']
        self.interpOCV = interp1d(self.SOC,self.OCV)
        self.interpSOC = interp1d(self.OCV,self.SOC)


    def getslope(self,z):
        sorted_SOC = self.SOC.sort_values()
        slopes = np.diff(self.OCV)/np.diff(self.SOC)
        for k in range(len(self.SOC)):
            if z > sorted_SOC.iloc[-1] or z < sorted_SOC.iloc[0]:
                raise ValueError('z is out of range')
            elif z == sorted_SOC.iloc[-1]:
                slope = slopes[-1]
                break
            elif z < sorted_SOC.iloc[k+1]:
                slope = slopes[k]
                break
        return slope


    def OCVfromSOC(self,z):
        v = self.interpOCV(z)
        return v
    

    def SOCfromOCV(self,v):
        z = self.interpSOC(v)
        return z

    def plot(self,color='b',linestyle='-'):
        plt.figure()
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
        self.A = np.array([[1., 0.],[0.,  np.exp(-dt/(self.R1*self.C1))]], dtype=float)
        self.B = np.array([[-self.coulombic_efficiency*dt/self.total_capacity],[1 - np.exp(-dt/(self.R1*self.C1))]],dtype=float)
        self.C = np.array([self.OCVcurve.getslope(self.z), -self.R1],dtype=float)
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