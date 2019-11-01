import numpy as np
import sys
sys.path.append('/home/lucas/Documents/Log_Analysis/Battery')
sys.path.append('/Users/Lucas/Documents/Travail/Yuneec/LogAnalysis')
import analog
import matplotlib.pyplot as plt


#%%
import platform 
if platform.system() == 'Linux':
    folder = '/home/lucas/Documents/Log_Analysis/Logs/Snow Orange (Battery 9) z0=1'
if platform.system() == 'Darwin':
    folder = '/Users/Lucas/Documents/Travail/Yuneec/Logs/Snow Orange (Battery 9) z0=1'
log_file = analog.pathfromQGC(folder,index=195)
info = analog.logextract(log_file,'battery_status')


#%%
u = info['battery_current']
u = np.reshape(u,(len(u),1))
print(np.shape(u))
t = info['time_bs']
y = info['battery_voltage']/4
y = np.reshape(y,(len(y),1))
print(np.shape(y))


#%%
u=u[t>60]
y=y[t>60]
t=t[t>60]


#%%
N = 14500
twoi = 14500 # i>n, the system order
i = int(twoi/2)
print(len(u))
print(i)


#%%
plt.figure()
plt.subplot(211)
plt.plot(t,y)
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('y')

plt.subplot(212)
plt.plot(t,u)
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('u')
plt.axvline(t[twoi+N-1],color='r')



# block Hankel control matrix

U = u[:twoi]

for n in range(N-1):
    #print(f'n={n}')
    U = np.hstack([U,u[n:twoi+n]])
    #print(U)


# block Hankel observation matrix

Y = y[:twoi]

for n in range(N-1):
    #print(f'n={n}')
    Y = np.hstack([Y,y[n:twoi+n]])
    #print(U)



#%%
Up = U[0:i,:] # past inputs
Uf = U[i:,:] # future inputs 
Yp = Y[0:i,:] # past outputs
Yf = Y[i:,:] # future outputs


#%%
# print(f'Up is {np.shape(Up)}')
# print(f'Yp is {np.shape(Yp)}')

Wp = np.vstack([Up,Yp])
# print(f'Uf is {np.shape(Uf)}')
# print(f'pinv(Uf*Uf.T) is {np.shape(np.linalg.pinv(Uf@Uf.T))}')
PiUf = Uf.T@np.linalg.pinv(Uf@Uf.T)@Uf
# print(f'Id is {np.shape(np.identity(np.shape(Uf)[0]))}')
# print(f'PiUf is {np.shape(PiUf)}')
PiUfortho = np.identity(np.shape(PiUf)[0]) - PiUf


#%%
# print(f'Yf is {np.shape(Yf)}')
# print(f'Wp is {np.shape(Wp)}') # should be 2i long
# print(f'PiUfortho is {np.shape(PiUfortho)}')
ksi = (Yf@PiUfortho)@np.linalg.pinv((Wp@PiUfortho))@Wp 
u,s,vh = np.linalg.svd(ksi,compute_uv=True)


#%%
# print(f'ksi is {np.shape(ksi)}')
# print(f'u is {np.shape(u)}')
# print(f's is {np.shape(s)}')
# print(f'vh is {np.shape(vh)}')
Sigma = np.hstack([np.diag(s),np.zeros((i,N-i))])
print(f'maximal error is {np.max(ksi-u@Sigma@vh)}')


#%%
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure()
plt.axhline(1,color='r',LineStyle='-.')
plt.semilogy(s,'x')
plt.grid()


print(f'Order to be retained is {np.shape(s[s>1])[0]}')
print(s[s>1])


