# %% codecell

# %% codecell
import pandas as pd
from datetime import datetime
import numpy as np
from dateutil.parser import parse
import matplotlib.pyplot as plt
import matplotlib.lines as line

import seaborn as sns

import statsmodels as sm

from math import sqrt

from scipy import stats
from scipy.odr import *
from scipy.integrate import odeint
from scipy.optimize import curve_fit

%matplotlib inline
# %% codecell
data=pd.read_csv('dpc-covid19-ita-andamento-nazionale.csv')
data['data'] = pd.to_datetime(data['data'])
data['data'] = data['data'].dt.strftime('%d/%m')
data.index=data['data']
data['totale_attualmente_positivi']=data['totale_attualmente_positivi']/60483973
data.drop('data',axis=1)
# %% codecell
fig=plt.figure(figsize=(18,16))
new=fig.add_subplot(1,1,1)
new.plot(data.index,data['nuovi_attualmente_positivi']+data['dimessi_guariti']+data['deceduti'],'--',label='NEW')
new.plot(data.index,data['nuovi_attualmente_positivi'],'--',label='ACNEW')
new.legend()

#grw=fig.add_subplot(2,2,3)
#grw.plot(data.index,data['GRW'],'--',label='GRW')
#grw.plot(data.index,data['ACGRW'],'--',label='ACGRW')
#l = line.Line2D([13,25], [1,1])
#grw.add_line(l)
#grw.legend()

fig
# %% codecell
trans = 2.3 #rateo di trasmissione
recov = 0.6 #rateo di recovery
tmax = 60#numero di giorni fittizio
totpos=data['totale_attualmente_positivi']
#initial conditions
sstart = 1-totpos['13/03']
rstart = 0
istart = data.loc['13/03']
# time points
t=data.index.to_series(range())
t
t_fit=t[18:]
t_fit
#fit base
def modelI(t,trans,recov):
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
    # solve ODE
    y = odeint(model,y0,t)
    return y[:,1]
# %% codecell
p,cov=curve_fit(modelI,t_fit,totpos['13/03':])
trans,recov=p
fit=modelI(t,trans,recov)
tot.plot(data.index,fit,'k--',label='Model')
tot.legend()
fig
# %% codecell
p
