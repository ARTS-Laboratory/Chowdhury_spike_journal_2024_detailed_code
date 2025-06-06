# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:35:34 2025

@author: chypu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter #
#%% plotting with specific parameters
plt.rcdefaults()
# Updating Parameters for Paper
params = {
    'lines.linewidth' :1,
    'lines.markersize' :0.5,
   'axes.labelsize': 8,
    'axes.titlesize':8,
    'axes.titleweight':'normal',
    'font.size': 8,
    'font.family': 'Times New Roman', # 'Times New RomanArial'
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
    'legend.shadow':'False',
   'legend.fontsize': 8,
   'xtick.labelsize':8,
   'ytick.labelsize':8,
   'text.usetex': False,
    'figure.autolayout': True,
   # 'figure.figsize': [3.5,2.5] # 6,4 for test3, ansys data 6,7
   'figure.figsize': [6.5,5] # 6,4 for test3, ansys data 6,7
   }
plt.rcParams.update(params)

#%% new plot with linestyle only
#%% Reading data
filename = "data/9_full_sensor_spike_data_processed.csv" 
D1=pd.read_csv(filename,skiprows=2)
t1=D1.iloc[:,0] # time in second
# t1=D1.iloc[:,0]/60 # time in minute
T1=D1.iloc[:,1] # spike 1 temperature
H1=D1.iloc[:,2] # spike 1 Humidity
P1=D1.iloc[:,3] # spike 1 Pressure(Pa)
C1=D1.iloc[:,4] # spike 1 TDS/conductivity (volts)
#%%
T2=D1.iloc[:,5] # spike 2 temperature
H2=D1.iloc[:,6] # spike 2 Humidity
P2=D1.iloc[:,7] # spike 2 Pressure(Pa)
C2=D1.iloc[:,8] # spike 2 TDS/conductivity (volts)
#%%
T3=D1.iloc[:,9] # spike 3 temperature
H3=D1.iloc[:,10] # spike 3 Humidity
P3=D1.iloc[:,11] # spike 3 Pressure(Pa)
C3=D1.iloc[:,12] # spike 3 TDS/conductivity (volts)
#%%
T4=D1.iloc[:,13] # spike 4 temperature
H4=D1.iloc[:,14] # spike 4 Humidity
P4=D1.iloc[:,15] # spike 4 Pressure(Pa)
C4=D1.iloc[:,16] # spike 4 TDS/conductivity (volts)
#%%
T5=D1.iloc[:,17] # spike 5 temperature
H5=D1.iloc[:,18] # spike 5 Humidity
P5=D1.iloc[:,19] # spike 5 Pressure(Pa)
C5=D1.iloc[:,20] # spike 5 TDS/conductivity (volts)
#%%
T6=D1.iloc[:,21] # spike 6 temperature
H6=D1.iloc[:,22] # spike 6 Humidity
P6=D1.iloc[:,23] # spike 6 Pressure(Pa)
C6=D1.iloc[:,24] # spike 6 TDS/conductivity (volts)
#%%
T7=D1.iloc[:,25] # spike 7 temperature
H7=D1.iloc[:,26] # spike 7 Humidity
P7=D1.iloc[:,27] # spike 7 Pressure(Pa)
C7=D1.iloc[:,28] # spike 7 TDS/conductivity (volts)
#%%
T8=D1.iloc[:,29] # spike 8 temperature
H8=D1.iloc[:,30] # spike 8 Humidity
P8=D1.iloc[:,31] # spike 8 Pressure(Pa)
C8=D1.iloc[:,32] # spike 8 TDS/conductivity (volts)
#%%
T9=D1.iloc[:,33] # spike 9 temperature
H9=D1.iloc[:,34] # spike 9 Humidity
P9=D1.iloc[:,35] # spike 9 Pressure(Pa)
C9=D1.iloc[:,36] # spike 9 TDS/conductivity (volts)
# Create a single figure

#%% Convert voltage to resistance  
#the sensor outputs a voltage between 0 and 2.3 V, which corresponds to a current range of 3 to 6 mA, I will start with a linear scaling approach to estimate the current from the voltage readings:
# I = ((V / 2.3) * (6 - 3)) + 3  // V in volts, I in mA
# The resistance by applying Ohm’s Law:
# R=V/I​
def voltage_to_resistance(V):
    I_mA = (V / 2.3) * 3 + 3  # current in mA
    I_A = I_mA / 1000         # convert to A
    R = V / I_A               # resistance in ohms
    return R

R1 = voltage_to_resistance(C1)
R2 = voltage_to_resistance(C2)
R3 = voltage_to_resistance(C3)
R4 = voltage_to_resistance(C4)
R5 = voltage_to_resistance(C5)
R6 = voltage_to_resistance(C6)
R7 = voltage_to_resistance(C7)
R8 = voltage_to_resistance(C8)
R9 = voltage_to_resistance(C9)

#%% Resistance/conductivity/TDS
fig1 = plt.figure()
ax = fig1.add_subplot(221)
ax.plot(t1, R1, linestyle='-', label='spike 1')
ax.plot(t1, R2, linestyle='--', label='spike 2')
ax.plot(t1, R3, linestyle='-.', label='spike 3')
ax.plot(t1, R4, linestyle=':', label='spike 4')
ax.plot(t1, R5, linestyle='-', label='spike 5')
ax.plot(t1, R6, linestyle='-', label='spike 6')
ax.plot(t1, R7, linestyle='-', label='spike 7')
ax.plot(t1, R8, linestyle='-', label='spike 8')
ax.plot(t1, R9, linestyle='-', label='spike 9')
ax.plot([1397.96805,1397.96805],[-1,55],'--',color='black',linewidth=1)# nonstationary
ax.plot([1853.32298,1853.32298],[-1,55],'--',color='black',linewidth=1)# nonstationary
ax.plot([2247.93314,2247.93314],[-1,55],'--',color='black',linewidth=1)# nonstationary
ax.plot([4706.0852,4706.0852],[-1,55],'--',color='black',linewidth=1)# nonstationary
ax.plot([6567.4691,6567.4691],[-1,55],'--',color='black',linewidth=1)# nonstationary
# ax.legend(loc='lower right',bbox_to_anchor=(0.95,0.01),ncol=1,facecolor='white', edgecolor = 'black', framealpha=1)

ax.set_xlim(500, 7417)
# ax.legend(loc='upper right', ncol=2, facecolor='white', edgecolor='black', framealpha=1)
# plt.title('Resistance vs. Time')
plt.xlim(500,7417)
plt.ylim(-1,55)
plt.xlabel('time (s)') #time
plt.ylabel(r'resistance ($\Omega$)')

#%% temperature
# fig1 = plt.figure()
ax = fig1.add_subplot(222)
ax.plot(t1,T1,linestyle='-', label='spike 1')
ax.plot(t1,T2,linestyle='--', label='spike 2')
ax.plot(t1,T3,linestyle='-.', label='spike 3')
ax.plot(t1,T4,linestyle=':',label='spike 4')
ax.plot(t1,T5,linestyle='-', label='spike 5')
ax.plot(t1,T6,linestyle='-', label='spike 6')
ax.plot(t1,T7,linestyle='-', label='spike 7')
ax.plot(t1,T8,linestyle='-', label='spike 8')
ax.plot(t1,T9,linestyle='-', label='spike 9')
ax.plot([1397.96805,1397.96805],[24.50,22.65],'--',color='black',linewidth=1)# nonstationary
ax.plot([1853.32298,1853.32298],[24.50,22.65],'--',color='black',linewidth=1)# nonstationary
ax.plot([2247.93314,2247.93314],[24.50,22.65],'--',color='black',linewidth=1)# nonstationary
ax.plot([4706.0852,4706.0852],[24.50,22.65],'--',color='black',linewidth=1)# nonstationary
ax.plot([6567.4691,6567.4691],[24.50,22.65],'--',color='black',linewidth=1)# nonstationary
plt.xlim(500,7417)
plt.ylim(22.65,24.50)
ax.legend(loc='lower right',bbox_to_anchor=(1,0.5),ncol=2,facecolor='white', edgecolor = 'black', framealpha=1)

plt.xlabel('time (s)') #sample points
# plt.ylabel('Temperature (degree celcius)')
plt.ylabel('temperature (°C)')
# plt.title('(b)',y=-1, ha='center')




#%% humidity
# fig1 = plt.figure()
ax = fig1.add_subplot(223)
ax.plot(t1,H1,linestyle='-', label='spike 1')
ax.plot(t1,H2,linestyle='--', label='spike 2')
ax.plot(t1,H3,linestyle='-.', label='spike 3')
ax.plot(t1,H4,linestyle=':',label='spike 4')
ax.plot(t1,H5,linestyle='-', label='spike 5')
ax.plot(t1,H6,linestyle='-', label='spike 6')
ax.plot(t1,H7,linestyle='-', label='spike 7')
ax.plot(t1,H8,linestyle='-', label='spike 8')
ax.plot(t1,H9,linestyle='-', label='spike 9')
ax.plot([1397.96805,1397.96805],[44.8,63.2],'--',color='black',linewidth=1)# nonstationary
ax.plot([1853.32298,1853.32298],[44.8,63.2],'--',color='black',linewidth=1)# nonstationary
ax.plot([2247.93314,2247.93314],[44.8,63.2],'--',color='black',linewidth=1)# nonstationary
ax.plot([4706.0852,4706.0852],[44.8,63.2],'--',color='black',linewidth=1)# nonstationary
ax.plot([6567.4691,6567.4691],[44.8,63.2],'--',color='black',linewidth=1)# nonstationary
plt.xlim(500,7417)
plt.ylim(44.8,63.2)
# ax.legend(loc='lower right',bbox_to_anchor=(0.95,0.01),ncol=1,facecolor='white', edgecolor = 'black', framealpha=1)
plt.xlabel('time (s)') #sample points
plt.ylabel('humidity (%)')

#%% pressure (pa)
# fig1 = plt.figure()
ax = fig1.add_subplot(224)
ax.plot(t1,P1,linestyle='-', label='spike 1')
ax.plot(t1,P2,linestyle='--', label='spike 2')
ax.plot(t1,P3,linestyle='-.', label='spike 3')
ax.plot(t1,P4,linestyle=':',label='spike 4')
ax.plot(t1,P5,linestyle='-', label='spike 5')
ax.plot(t1,P6,linestyle='-', label='spike 6')
ax.plot(t1,P7,linestyle='-', label='spike 7')
ax.plot(t1,P8,linestyle='-', label='spike 8')
ax.plot(t1,P9,linestyle='-', label='spike 9')

ax.plot([1397.96805,1397.96805],[100185,100415],'--',color='black',linewidth=1)# nonstationary
ax.plot([1853.32298,1853.32298],[100185,100415],'--',color='black',linewidth=1)# nonstationary
ax.plot([2247.93314,2247.93314],[100185,100415],'--',color='black',linewidth=1)# nonstationary
ax.plot([4706.0852,4706.0852],[100185,100415],'--',color='black',linewidth=1)# nonstationary
ax.plot([6567.4691,6567.4691],[100185,100415],'--',color='black',linewidth=1)# nonstationary
plt.xlim(500,7417)
plt.ylim(100185,100415)
# ax.legend(loc='lower right',bbox_to_anchor=(0.95,0.01),ncol=1,facecolor='white', edgecolor = 'black', framealpha=1)

plt.xlabel('time (s)') #sample points
plt.ylabel('pressure (Pa)')
# plt.title('(b)',y=-1, ha='center')

plt.savefig('plots/9spikes_resis_2D_plot.pdf', dpi=100)
plt.savefig('plots/9spikes_resis_2D_plot.png', dpi=100)
