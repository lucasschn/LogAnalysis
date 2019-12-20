import numpy as np
import pandas as pd
from scipy.interpolate import interp1d as interp1d
import matplotlib.pyplot as plt
import datetime
from analog import logextract
from os import listdir
from scipy.stats import chi2
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

        self.poly_coeff = np.polyfit(self.SOC,self.OCV,11)
        print(f'Polynomial coefficients are {self.poly_coeff}')

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
        self.interpOCV = interp1d(self.SOC,self.OCV)
        self.interpSOC = interp1d(self.OCV,self.SOC)

    def intrescorr(self,R0):
        ''' Corrects OCV curve with internal resistance R0.'''
        if self.mode == 'charge':
            self.OCV = self.OCV - self.current*R0
        elif self.mode == 'discharge':
            self.OCV = self.OCV + self.current*R0
        

    def getslope(self,z):
        slope = (self.OCVfromSOC(min(1.0,z+0.01))-self.OCVfromSOC(max(1e-3,z-0.01)))/(min(1.0,z+0.01)-max(1e-3,z-0.01))
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
 

    def OCVfromSOC(self,SOC):
        #OCV = self.interpOCV(SOC)
        OCV = 0
        for poly_expo  in range(len(self.poly_coeff)):
            OCV += self.poly_coeff[len(self.poly_coeff) - 1 - poly_expo] * pow(SOC, poly_expo)
        return OCV
    

    def SOCfromOCV(self,v):
        z = self.interpSOC(v)
        return z

    def plot(self,color='b',linestyle='-'):
        plt.plot(self.SOC,self.OCV,color=color,linestyle=linestyle)
        plt.xlabel('State of Charge (-)')
        plt.ylabel('Open Circuit Voltage (V)')
        plt.grid()

class DynParamOptimizer:

    def __init__(self,batname):
        folder = '/home/lucas/Documents/Log_Analysis/Logs'
        for subfolder in listdir(folder):
            # print(subfolder)
            # print(subfolder.find(batname))
            if subfolder.find(batname) > 0:
                batsubfolder = subfolder
                print(batsubfolder)
        
        path = f'{folder}/{batsubfolder}'
        if len(listdir(path)) == 1:
            
            self.logfile = f'{folder}/{batsubfolder}/{listdir(path)[0]}'
        else :
            print('The data is splitted in several logs, which is not handled yet.')


    def iofromlog(self):
        info = logextract(self.logfile,'battery_status')
        y = info['battery_voltage']
        u = info['battery_current']
        return y,u

    def optimizeRC(self):
        y,u = self.iofromlog()
        ny = len(y)
        n = 2 # 1 pair of RC

        A = self.SISOsubid(y,u,n)
        print(A)

        # self.Battery.R1 = R1opt
        # self.Battery.C1 = C1op1

    def SISOsubid(self,y, u, n):
        """
        Identify state-space "A" matrix from input-output data.
        y: vector of measured outputs
        u: vector of measured inputs
        n: number of poles in solution
        A: discrete-time state-space state-transition matrix.

        Theory from "Subspace Identification for Linear Systems Theory - Implementation
        - Applications" Peter Van Overschee / Bart De Moor (VODM) Kluwer Academic
        Publishers, 1996. Combined algorithm: Figure 4.8 page 131 (robust). Robust
        implementation: Figure 6.1 page 169.

        Code adapted from "subid.m" in "Subspace Identification for Linear Systems"
        toolbox on MATLAB CENTRAL file exchange, originally by Peter Van Overschee,
        Dec. 1995
        """

        ny = len(y)
        i = 2*n
        twoi = 4*n

        # Determine the number of columns in the Hankel matrices
        j = ny - twoi + 1

        # Make Hankel matrices Y and U
        Y = np.zeros((twoi, j))
        U = np.zeros((twoi, j))

        for k in range(2*i):
            Y[k] = y[k:k+j]
            U[k] = u[k:k+j]

        print(f'U = {U}, shape {np.shape(U)}')
        print(f'Y = {Y}, shape {np.shape(Y)}')

        # Compute the R factor
        UY = np.concatenate((U, Y))     # combine U and Y into one array
        q, r = np.linalg.qr(UY.T)       # QR decomposition
        R = r.T                         # transpose of upper triangle
        print(f'R = {R}')
        # STEP 1: Calculate oblique and orthogonal projections
        # ------------------------------------------------------------------

        Rf = R[-i:]                                 # future outputs
        print(f'Rf = {Rf}')
        Rp = np.concatenate((R[:i], R[2*i:3*i]))    # past inputs and outputs
        Ru = R[i:twoi, :twoi]                       # future inputs

        RfRu = np.linalg.lstsq(Ru.T, Rf[:, :twoi].T)[0].T
        RfRuRu = RfRu.dot(Ru)
        tm1 = Rf[:, :twoi] - RfRuRu
        tm2 = Rf[:, twoi:4*i]
        Rfp = np.concatenate((tm1, tm2), axis=1)    # perpendicular future outputs

        RpRu = np.linalg.lstsq(Ru.T, Rp[:, :twoi].T)[0].T
        RpRuRu = RpRu.dot(Ru)
        tm3 = Rp[:, :twoi] - RpRuRu
        tm4 = Rp[:, twoi:4*i]
        Rpp = np.concatenate((tm3, tm4), axis=1)    # perpendicular past inputs and outputs

        # The oblique projection is computed as (6.1) in VODM, page 166.
        # obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * (Wp/Ufp)
        # The extra projection on Ufp (Uf perpendicular) tends to give
        # better numerical conditioning (see algo on VODM page 131)

        # Funny rank check (SVD takes too long)
        # This check is needed to avoid rank deficiency warnings

        nmRpp = np.linalg.norm(Rpp[:, 3*i-3:-i], ord='fro')
        if nmRpp < 1e-10:
            # oblique projection as (Rfp*pinv(Rpp')') * Rp
            Ob = Rfp.dot(np.linalg.pinv(Rpp.T).T).dot(Rp)
        else:
            # oblique projection as (Rfp/Rpp) * Rp
            Ob = (np.linalg.lstsq(Rpp.T, Rfp.T)[0].T).dot(Rp)

        # STEP 2: Compute weighted oblique projection and its SVD
        #         Extra projection of Ob on Uf perpendicular
        # ------------------------------------------------------------------

        ObRu = np.linalg.lstsq(Ru.T, Ob[:, :twoi].T)[0].T
        ObRuRu = ObRu.dot(Ru)
        tm5 = Ob[:, :twoi] - ObRuRu
        tm6 = Ob[:, twoi:4*i]
        WOW = np.concatenate((tm5, tm6), axis=1)

        U, S, _ = np.linalg.svd(WOW, full_matrices=False)
        ss = np.diag(S)

        # STEP 3: Partitioning U into U1 and U2 (the latter is not used)
        # ------------------------------------------------------------------

        U1 = U[:, :n]       # determine U1

        # STEP 4: Determine gam = Gamma(i) and gamm = Gamma(i-1) (Gamma is the observablity matrix)
        # ------------------------------------------------------------------

        gam = U1 * np.diag(np.sqrt(ss[:n]))
        print(f'gam = {gam}')
        gamm = gam[:i-1,:]
        print(f'pinv(gam) = {np.linalg.pinv(gam)}')
        gam_inv = np.linalg.pinv(gam)[0]          # pseudo inverse of gam
        gamm2 = np.array([[gamm], [gamm]])
        gamm_inv = np.linalg.pinv(gamm2)[0][0]*2    # pseudo inverse of gamm
        print(f'gam_inv = {gam_inv}')
        # STEP 5: Determine A matrix (also C, which is not used)
        # ------------------------------------------------------------------
        print(gam_inv.dot(R[-i:,:-i]))
        print(np.zeros(n))
        tm7 = np.concatenate((gam_inv.dot(R[-i:, :-i]), np.zeros(n)))
        print(f'tm7 = {tm7}, shape {np.shape(tm7)}')
        tm8 = R[i:twoi, 0:3*i+1]
        print(f'tm8 = {tm8}, shape {np.shape(tm8)}')
        Rhs = np.vstack((tm7, tm8))
        Lhs = np.vstack((gamm_inv*R[-i+1, :-i+1], R[-i, :-i+1]))
        sol = np.linalg.lstsq(Rhs.T, Lhs.T)[0].T    # solve least squares for [A; C]
        A = sol[:n-1, :n-1]                           # extract A

        return A


   


class Battery:

    def __init__(self,z,Q,eta=1):
        self.total_capacity = Q
        self.charge_state = z
        self.coulombic_efficiency = eta


    def setOCV(self,OCVcurve):
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
        self.A = np.array([[1.0, 0.],[0.,  np.exp(-dt/(self.R1*self.C1))]], dtype=float)
        self.B = np.array([[-self.coulombic_efficiency*dt/self.total_capacity],[1 - np.exp(-dt/(self.R1*self.C1))]],dtype=float)
        self.C = np.reshape(np.array([self.OCVcurve.getslope(self.z), -self.R1],dtype=float),(1,2))
        self.D = np.array([-self.R0],dtype=float)

    def varsim(self,time,current,curve=[]):
        ''' Simulation with variable stepsize'''

        self.varsimt = time
        self.varsimi = current
        
        if OCVcurve!=[]:
            self.OCVcurve = curve
        
        self.varsimx = np.array([[self.z],[0]])
        self.varsimv = []
        self.varsimdt = [np.mean(np.diff(self.varsimt))] + list(np.diff(self.varsimt))

        for k in range(len(self.lsimt)):
            self.statespace(self.varsimdt[k])
            if k ==0 :
                self.varsimx = np.concatenate([self.varsimx,self.A@np.reshape(self.varsimx[:,k],(2,1))+self.B*self.varsimi[k]],axis=1)
                self.varsimv.append(self.OCVcurve.OCVfromSOC(self.varsimx[0,k])-self.R0*self.varsimi[k]-self.R1*self.varsimx[1,k])
            else : 
                self.varsimx = np.concatenate([self.varsimx,self.A@np.reshape(self.varsimx[:,k],(2,1))+self.B*self.varsimi[k]],axis=1)
                self.varsimv.append(self.OCVcurve.OCVfromSOC(self.varsimx[0,k])-self.R0*self.varsimi[k]-self.R1*self.varsimx[1,k])
    
        return self.varsimv

    def lsim(self,time,current,OCVcurve,plot=False):

        self.lsimt = time
        self.lsimi = current
        self.lsimdt = np.mean(np.diff(self.simt))

        self.lsimx = np.array([[self.z],[0]])

        self.lsimv = np.array([])             
        for k in range(len(self.lsimt)):
            
            if k == 0:
                self.lsimx = np.concatenate([self.lsimx,self.A@np.reshape(self.lsimx,(2,1)) + self.B*self.lsimi[k]],axis=1)
                self.lsimv = self.C@np.reshape(self.lsimx[:,k],(2,1)) + self.D*self.lsimi[k]
            else:
                self.lsimx = np.concatenate([self.lsimx,self.A@np.reshape(self.lsimx[:,k],(2,1)) + self.B*self.lsimi[k]],axis=1)
                self.lsimv = np.concatenate([self.lsimv,self.C@np.reshape(self.lsimx[:,k],(2,1)) + self.D*self.lsimi[k]])

        if plot :
            plt.figure()
            plt.subplot(211)
            plt.plot(time,self.lsimv,label='lin. sim.')
            plt.legend()
            plt.grid()
            plt.ylabel('Cell voltage (V)')
            plt.subplot(212) 
            plt.plot(time,current)
            plt.xlabel('time (s)')
            plt.ylabel('Current (A)')
            plt.grid()

        return self.lsimv

    def reset(self,R0,R1,C1,z):
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1   
        self.z = z     

    def simulate(self,time,current,plot=False):
        self.simt = time
        self.simi = current
        self.simdt = np.mean(np.diff(self.simt))

        self.simz = []
        self.simz.append(self.z) 
        self.simi1 = [] 
        self.simi1.append(0)

        self.simv = []             
        for k in range(len(self.simt)):
            self.simz.append(self.simz[k] - self.coulombic_efficiency*self.simdt/self.total_capacity * self.simi[k])
            self.simi1.append(np.exp(-self.simdt/(self.R1*self.C1)) * self.simi1[k] + (1 - np.exp(-self.simdt/(self.R1*self.C1)))*self.simi[k])
            self.simv.append(self.OCVcurve.OCVfromSOC(self.simz[k]) - self.R1*self.simi1[k] - self.R0*self.simi[k])

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


    def kfinit(self,covw,covx0,alpha=0.1):
        self.xhat = np.array([[self.z],[0.]]) # is a stack of 2x1 arrays = a 2xk array
        if covx0==[]:
            self.covx = np.array([[1e-4,0.],[0., .1]]) # is a stack of 2x2 arrays
        else:
            self.covx = covx0

        if covw==[]:
            self.covw = np.array([[0.05, 0.],[0., 0.]]) # is a constant 2x2 array
        else:
            self.covw = covw

        self.covv = 0.5017 + 0.5 # is a constant 1x1 array
        self.chisquare = chi2.ppf(1-alpha,df=1)  # upper bound for nees based on Chi2 test, with m degrees of freedom and 1-alpha confidence
        self.u = 0 # is a stack of 1x1 arrays = a 1-D array
        self.yhat = self.OCVcurve.OCVfromSOC(self.xhat[0]) - self.R1*self.xhat[1]

    def kfupdate(self,u,y):

        #1b
        self.covx = self.A@self.covx@self.A.T + self.covw
        #1c
        self.yhat = self.OCVcurve.OCVfromSOC(self.xhat[0]) - self.R1*self.xhat[1] + self.D*u
        #2a

        # Voltage sensor fault detection:
        covxy = np.reshape(self.covx@self.C.T,(2,1))
        covy = self.C@self.covx@self.C.T + self.covv
        
        #1a & #2b
        inno = y - self.yhat

        # if inno > 0.05:
        #     inno = 0.05
        # elif inno < -0.05:
        #     inno = -0.05

        nees = inno*1/covy*inno # normalized estimation error squared
        if nees < self.chisquare:
            self.L = np.reshape(covxy/covy,(2,1))
        else:
            self.L = np.reshape([0,0],(2,1))

        self.xhat = np.reshape(self.A@self.xhat + self.B*u + self.L*inno,(2,1))

        if self.xhat[0] > 1.0:
            self.xhat[0] = 1.0
        elif self.xhat[0] < 0.0:
            self.xhat[1] = 0.0

        #2c
        tmp1 = (np.identity(2) - self.L@self.C)
        tmp2 = (np.identity(2) - self.L@self.C).T
        tmp3 = self.L@np.reshape(self.covv,(1,1))@self.L.T
        self.covx = tmp1@self.covx@tmp2 + tmp3  

    def kfrun(self,t,u,y,covw=[],covx0=[],alpha=0.05):
        self.statespace(t[1]-t[0])
        self.kfinit(covw,covx0,alpha=alpha)
        xhat = self.xhat
        covx = self.covx
        yhat = self.yhat
        L =  np.reshape(self.covx@self.C.T/(self.C@self.covx@self.C.T + self.covv),(2,1))
        # equivalent to updateStatus
        for k in range(1,len(u)):
            self.statespace(t[k]-t[k-1])
            self.kfupdate(u[k],y[k])
            xhat = np.concatenate([xhat,self.xhat],axis=1)
            covx =  np.dstack([covx,self.covx])
            yhat = np.concatenate([yhat,self.yhat])
            L = np.concatenate([L,self.L],axis=1)
        return xhat,covx,yhat,L