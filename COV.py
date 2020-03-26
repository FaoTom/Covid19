# %% codecell

# %% codecell
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as line

import seaborn as sns

import statsmodels as sm

from math import sqrt

from scipy import stats
from scipy.odr import *
from scipy.integrate import odeint

%matplotlib inline
# %% codecell
xlsx=pd.ExcelFile('COV.xlsx')
data=pd.read_excel(xlsx,'Foglio 1')
data.columns=['Giorno','TOT','NEW','GRW','ACTOT','ACNEW','ACGRW']
data.columns.name='Day'
data.dropna(how='all',inplace=True)
data.drop('Giorno',axis=1,inplace=True)
data.index=range(12,26)
data.drop(12,inplace=True)
data
# %% codecell
fig=plt.figure(figsize=(18,16))
fig.subplots_adjust(wspace=0.1,hspace=0)
tot=fig.add_subplot(2,2,1)
new=fig.add_subplot(2,2,2)
grw=fig.add_subplot(2,2,3)
tot.scatter(data.index,data['TOT'],label='TOT')
tot.scatter(data.index,data['ACTOT'],label='ACTOT')
tot.legend()
new.plot(data.index,data['NEW'],'--',label='NEW')
new.plot(data.index,data['ACNEW'],'--',label='ACNEW')
new.legend()
grw.plot(data.index,data['GRW'],'--',label='GRW')
grw.plot(data.index,data['ACGRW'],'--',label='ACGRW')
l = line.Line2D([13,25], [1,1])
grw.add_line(l)
grw.legend()
# %% codecel

trans = 2.3 #rateo di trasmissione
recov = 0.23 #rateo di recovery
tmax = 50 #numero di giorni fittizio

#initial conditions
sstart = 0.99
rstart = 0
istart = 0.01

# function that returns dy/dt
def model(y,t):
    S=y[0]
    I=y[1]
    R=y[2]
    dS = -trans*S*I
    dI = trans*S*I-recov*I
    dR = recov*I
    return [dS,dI,dR]
y0=[sstart,istart,rstart]
# time points
t = np.linspace(0,tmax)
# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()

#==================================================

def f(y,t):
    Xi = y[0]
    Yi = y[1]
    Zi = y[2]
    f0 = Xi*(onr/(Xi**3) + (H**2)*(Zi**2)/(2.0*rho_c) + (v0*np.exp(-l*Yi*k))/(rho_c))**(1.0/2.0)
    f1 =  Zi
    f2 =  -3*Zi*(H**2)*(onr/(Xi**3) + (H**2)*(Zi**2)/(2.0*rho_c) + (v0*np.exp(-l*Yi*k))/(rho_c))**0.5 + (l*k*v0*np.exp(-l*Yi*k))/(H**2)
    return [f0,f1,f2]

X0 = [1.0]
Y0 = [1.0]
Z0 = [c]
y0 = [X0,Y0,Z0]
t = np.linspace(start=1.0,stop=0.0,num=10001)

soln = odeint(f,y0,t)
X = soln[:,0]
Y = soln[:,1]
Z = soln[:,2]
