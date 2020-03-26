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
