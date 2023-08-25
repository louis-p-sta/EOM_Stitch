# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24, 2023 @ 12:07

@author: louis
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from lmfit import Minimizer, fit_report, Parameters, report_fit, conf_interval
from scipy.optimize import fsolve
#from scipy.optimize import curve_fit
#from sklearn.linear_model import LinearRegression

def normalize(data):
    data = np.subtract(data, np.max(data))
    data = np.divide(data,np.min(data))
    return data
#Data lives in the DispersionTest folder - did not want to make a second copy of it since it's large.
sweep1_cavity = pd.read_csv("C:/Users/louis\OneDrive - University of Ottawa/uOttawa/MPL Docs/Code/DispersionTest/Data/FinalData/#1backup/Ch2.csv",  header = None)
sweep1_HCN = pd.read_csv("C:/Users/louis/OneDrive - University of Ottawa/uOttawa\MPL Docs\Code\DispersionTest\Data\FinalData\#1backup\Ch1.csv",  header = None) #192.7385 THz
sweep1_cavity = sweep1_cavity.iloc[:,0]
sweep1_HCN = sweep1_HCN.iloc[:,0]
sweep2_cavity = pd.read_csv("C:/Users/louis/OneDrive - University of Ottawa/uOttawa\MPL Docs\Code\DispersionTest\Data\FinalData\#2\Ch2.csv",  header = None)
sweep2_HCN = pd.read_csv("C:/Users/louis/OneDrive - University of Ottawa/uOttawa\MPL Docs\Code\DispersionTest\Data\FinalData\#2\Ch1.csv", header = None) #192.6328 THz
sweep2_cavity = sweep2_cavity.iloc[:,0]
sweep2_HCN = sweep2_HCN.iloc[:,0]
#calibration = pd.read_csv("C:/Users/louis/OneDrive - University of Ottawa/uOttawa\MPL Docs\Code\DispersionTest\Data\Reference_Cavity_1550nm.csv") #Yes
#x_frequencies = calibration.iloc[:,0]
#HCN_calibration = calibration.iloc[:,1]
#cavity_calibration = calibration.iloc[:,2]

# #%%
# HCN_calibration_normalized = normalize(HCN_calibration)
# HCN_peaks1 = find_peaks(HCN_calibration_normalized, distance = 10000, prominence = 0.1, plateau_size = 100)[0] #Order when you come back
# HCN_peaks2 = find_peaks(HCN_calibration_normalized, distance = 10000, prominence = 0.1)[0]
# n=0
# delete = []
# for peak in HCN_peaks2:
#     if any(abs(np.subtract(HCN_peaks1, peak)) < 10000):
#         delete.append(n)
#     n+=1
#     #YAGNI
# HCN_peaks22 = np.delete(HCN_peaks2, delete)
# #HCN_peaks2 = np.delete(HCN_peaks2, np.where(HCN_peaks == HCN_peaks2)) #If this stops working, compare element by element using for loop
# concatenate = np.concatenate((HCN_peaks1,HCN_peaks22))
# HCN_peaks = sorted(concatenate)
# HCN_peak_values = HCN_calibration_normalized.iloc[HCN_peaks]
# x_HCN_peaks = x_frequencies[HCN_peaks]
# #You aren't going to need this cell of code...
# diffs = abs(np.subtract(x_HCN_peaks, 195.2360e12)) #Sweep 1 (R10) - always high to low
# first_peak = np.where(diffs == diffs.min())[0][0]
# diffs = abs(np.subtract(x_HCN_peaks, 195.1625e12)) #Sweep 2 (R9)
# second_peak = np.where(diffs == diffs.min())[0][0]
#%%
# fig3, ax3 = plt.subplots()
# ax3.set_title("HCN peaks")
# ax3.set_xlabel("Frequency (Hz)")
# ax3.set_ylabel("Normalized intensity")
# ax3.plot(x_frequencies, HCN_calibration_normalized)
# ax3.plot(x_frequencies[HCN_peaks],HCN_calibration_normalized[HCN_peaks],'o')
# ax3.plot(x_HCN_peaks.iloc[[first_peak,second_peak]], HCN_peak_values.iloc[[first_peak, second_peak]], 'o')
# #Count the number of FSRs in order to measure... need a file with the calibration?
#%%
# window = 1000
# HCN_pd_peaks = HCN_calibration_normalized.index[HCN_peaks]
# first_peak_pd = HCN_pd_peaks[first_peak]
# second_peak_pd = HCN_pd_peaks[second_peak]
# cavity_calibration = cavity_calibration.iloc[HCN_peaks[first_peak - 1]:HCN_peaks[second_peak + 1 ]]
# HCN_calibration_normalized = HCN_calibration_normalized.iloc[HCN_peaks[first_peak - 1]:HCN_peaks[second_peak + 1 ]]
# x_frequencies = x_frequencies.iloc[HCN_peaks[first_peak - 1]:HCN_peaks[second_peak + 1 ]]
#%%
# cavity_calibration_normalized = normalize(cavity_calibration)
# cavity_peaks = find_peaks(cavity_calibration_normalized, prominence = 0.5)[0]
# cavity_peaks_pd = cavity_calibration_normalized.index[cavity_peaks]
# x_cavity_peaks = x_frequencies[cavity_peaks_pd]
# #%%
# fig4, ax4 = plt.subplots(3,1, sharex = 'col')
# ax4[0].set_title("HCN calibration")
# ax4[1].set_title("Cavity resonances")
# ax4[1].plot(x_frequencies, cavity_calibration_normalized, "o-", markersize = 1)
# ax4[1].plot(x_frequencies.iloc[cavity_peaks], cavity_calibration_normalized.iloc[cavity_peaks],'o', markersize = 1)
# ax4[0].plot(x_frequencies,HCN_calibration_normalized, "o-", markersize = 1)
#%%
#Finding initial resonance
# diffs_init = abs(np.subtract(cavity_peaks_pd,first_peak_pd))
# index_init = np.where(diffs_init == diffs_init.min())[0][0]
# initial_resonance_pd = cavity_peaks_pd[index_init]

# #This assumes knowledge of the second peak. Now want to determine that without.
# diffs_final = abs(np.subtract(cavity_peaks_pd,second_peak_pd))
# index_final = np.where(diffs_final == diffs_final.min())[0][0]
# final_resonance_pd = cavity_peaks_pd[index_final]
# ax4[1].plot(x_frequencies[initial_resonance_pd],cavity_calibration_normalized[initial_resonance_pd],'o')
# ax4[1].plot(x_frequencies[final_resonance_pd], cavity_calibration_normalized[final_resonance_pd],'o')
#%% Calculate frequency between peaks
# num_of_FSRs = index_final - index_init - 1 #The data is imported from high to low frequency. So the final resonance, even though it is lower in frequency, does come later than the initial one.
# peaks = cavity_peaks_pd[index_init:index_final] #Since we work in wavelength, the leftmost resonance on the graph is actually the final one (same comment as above).
# n=0
# FSRs = []
# while n < len(peaks) - 1:
#     FSR = x_cavity_peaks[peaks[n]] - x_cavity_peaks[peaks[n+1]]
#     FSRs.append(FSR)
#     n+=1
# counted_FSRs = len(FSRs)
# if counted_FSRs != num_of_FSRs:
#     print("Something went wrong with the counting of FSRs.")
# frequ_between_peaks = np.sum(FSRs)
# actual_frequ_between_peaks = x_HCN_peaks[first_peak_pd] - x_HCN_peaks[second_peak_pd]
#FSR is about 118 MHz...
#%%
#Plot dispersion curve
# dispersion_x = x_cavity_peaks[peaks]
# dispersion_x = dispersion_x[0:len(dispersion_x)-1]
# ax4[2].set_title("C-Band dispersion curve")
# ax4[2].plot(dispersion_x, FSRs,'o-', markersize = 1)
#%% Higher frequency sweep
sweep1_cavity = normalize(sweep1_cavity[9000000:12055000])
sweep1_HCN = normalize(sweep1_HCN[9000000:12055000])
x1 = range(len(sweep1_cavity))
#%% Lower frequency
sweep2_cavity = normalize(sweep2_cavity[2295000:5750000])
sweep2_HCN = normalize(sweep2_HCN[2295000:5750000])
x2 = range(len(sweep2_cavity))
#%% Plot for slicing
fig0, ax0 = plt.subplots()
ax0.plot(range(len(sweep1_cavity)),sweep1_cavity)
ax0.plot(range(len(sweep1_HCN)),sweep1_HCN)
fig00, ax00 = plt.subplots()
ax00.plot(range(len(sweep2_cavity)),sweep2_cavity)
ax00.plot(range(len(sweep2_HCN)),sweep2_HCN)
#%% Finding peaks - solely for the purpose of counting
#first_peak_frequ = x_HCN_peaks[first_peak_pd]
#second_peak_frequ = x_HCN_peaks[second_peak_pd]
def sweepLength(HCN_peak, sweep_cavity, direction = "high-low", scrap_resonances = 0): #This function seems to repeat itself a little bit.
    full_sweep_cavity_peaks = find_peaks(sweep_cavity, prominence = 0.5, distance = 100)[0]
    diffs_sweep = abs(np.subtract(full_sweep_cavity_peaks,HCN_peak)) #Need to find index of FSR peak closed to HCN line.
    init_peak = np.where(diffs_sweep == diffs_sweep.min())[0][0]
    indexinit_sweep = full_sweep_cavity_peaks[init_peak]
    if direction == "high-low":
        ITLA_FSRs = len(full_sweep_cavity_peaks[init_peak:]) - scrap_resonances #Might need a minus one here, TBD.
        #sweep_peaks = peaks[:ITLA_FSRs]
        sweep_cavity_peaks = full_sweep_cavity_peaks[init_peak:]
    elif direction == "low-high":
        ITLA_FSRs = len(full_sweep_cavity_peaks[:init_peak]) - scrap_resonances
        #sweep_peaks = peaks[len(peaks) - ITLA_FSRs:]
        sweep_cavity_peaks = full_sweep_cavity_peaks[:init_peak]
    #x_sweep_peaks = x_cavity_peaks[sweep_peaks]
    #sweep_length = x_sweep_peaks.iloc[0] - x_sweep_peaks.iloc[-1]
    sweep_length = len(sweep_cavity_peaks)
    if sweep_length < 0:
        print("Error in sweep direction.")
    return HCN_peak, indexinit_sweep, init_peak, sweep_cavity_peaks, sweep_length, ITLA_FSRs, full_sweep_cavity_peaks  
HCN_peak_1 = find_peaks(sweep1_HCN, prominence = 0.5, plateau_size = 1000, distance = 100000)[0]
#ax0.plot(HCN_peak_1, sweep1_HCN.iloc[HCN_peak_1],'o', markersize = 1.5)
ITLA_first_peak, indexinit_sweep1, init_peak, sweep1_cavity_peaks, sweep1_length, ITLA_FSRs1, full_sweep_cavity_peaks_1 = sweepLength(HCN_peak_1, sweep1_cavity, direction = "high-low")
HCN_peak_2 = find_peaks(sweep2_HCN, prominence = 0.5, plateau_size = 1000, distance = 100000)[0]
#ax00.plot(HCN_peak_2, sweep2_HCN.iloc[HCN_peak_2],'o', markersize = 1.5)
ITLA_second_peak, indexinit_sweep2, final_peak, sweep2_cavity_peaks, sweep2_length, ITLA_FSRs2, full_sweep_cavity_peaks_2 = sweepLength(HCN_peak_2, sweep2_cavity, direction = "low-high")
#%% Finding EOM peaks for sweeps 1 & 2
def find_EOM1(sweep):
    EOM_peaks = find_peaks(sweep, prominence = [0.05,0.5], plateau_size = 1, width = 100, distance = 500)[0] #Try to refine positions with plateau size.
    return EOM_peaks
def find_EOM2(sweep):
    EOM_peaks = find_peaks(sweep, prominence = [0.05,0.5], plateau_size = 1, width = 100, distance = 500)[0]
    return EOM_peaks
sweep1_EOM_ix = find_EOM1(sweep1_cavity) 
sweep2_EOM_ix = find_EOM2(sweep2_cavity)
#%%

#%% Fitting the HCN peaks to a Lorentzian
# fitting_range = int(3e5)
# from scipy.constants import pi
# def lorentzian(x, p):
#     return p['amplitude']*(0.5*p["gamma"])**2/((x - p["center"])**2 + (0.5*p["gamma"])**2) + p["offset"]
# def fcn2minfit(params,x,y):
#     return lorentzian(x, params) - y
# HCN_data = sweep1_HCN
# HCN_peak = HCN_peak_1
# HCN_fit = HCN_data.iloc[0:fitting_range]
# #print(HCN_fit)
# HCN_fit_y = np.array(HCN_fit) #This has a pandas index
# #print(HCN_fit_y, len(HCN_fit_y))
# HCN_fit_x = x1[0:fitting_range]
# #print(HCN_fit_x, len(HCN_fit_x))
# #Try using lmfit
# params = Parameters()
# params.add('amplitude', value = 1)
# params.add('gamma', value = 1)
# params.add('center', value = HCN_peak[0])
# params.add('offset', value = 0.1)
# minner = Minimizer(fcn2minfit,params, fcn_args = (HCN_fit_x,HCN_fit_y))
# result1 = minner.minimize(method = "leastsq")
# report_fit(result1)
# report1 = fit_report(result1)
# fitted_y_HCN = lorentzian(HCN_fit_x,result1.params)
"""------------------------------------------------"""
# def fit_HCN(HCN_data, HCN_peak, fitting_range): #Number of sample points, in total, to encase the peak
#     HCN_fit = HCN_data.iloc[int(HCN_peak - fitting_range/2): int(HCN_peak + fitting_range/2)]
#     #print(HCN_fit)
#     HCN_fit_y = HCN_fit.tolist() #This has a pandas index
#     print(HCN_fit_y, len(HCN_fit_y))
#     HCN_fit_x = HCN_fit.index.tolist()
#     print(HCN_fit_x, len(HCN_fit_x))
#     #Try using lmfit
#     params = Parameters()
#     params.add('amplitude', value = 1)
#     params.add('gamma', value = 1)
#     params.add('center', value = HCN_peak)
#     minner = Minimizer(fcn2minfit,params, fcn_args = (HCN_fit_x,HCN_fit_y))
#     result = minner.minimize()
#     report_fit(result)
#     report = fit_report(result)
#     fitted_y_HCN = lorentzian(HCN_fit_x,result.params)
#     variables = (HCN_fit_y, HCN_fit_x)
#     return fitted_y_HCN, result.params, variables
# fitted_y_HCN_1, params_1, variables = fit_HCN(sweep1_HCN,HCN_peak_1, fitting_range)
#%%
#Check if FSRs seem consistent (follow similar pattern - fit and residuals) - do both mini graphs separately (scan speed somewhat constant?)
def test_FSR(sweep1_cavity_peaks):
    num_of_FSRs_ix = len(sweep1_cavity_peaks) - 1
    FSRs_ix = []
    n = 0
    while n < (len(sweep1_cavity_peaks) - 1):
        FSR_ix = sweep1_cavity_peaks[n+1] - sweep1_cavity_peaks[n]
        FSRs_ix.append(FSR_ix)
        n+=1
    counted_FSRs_ix = len(FSRs_ix)
    if counted_FSRs_ix != num_of_FSRs_ix:
        print("Something went wrong with the counting of FSRs.")
    length = len(sweep1_cavity_peaks) - 1
    FSRs_ix_x = sweep1_cavity_peaks[0:length]
    return FSRs_ix, FSRs_ix_x
FSRs_ix, FSRs_ix_x = test_FSR(sweep1_cavity_peaks)
FSRs_ix2, FSRs_ix_x2 = test_FSR(sweep2_cavity_peaks)
fig6, ax6 = plt.subplots(1,2, sharey = True)
ax6[0].set_title("FSR graph from ITLA, sweep 1")
ax6[0].set_xlabel("FSR location (sample points)")
ax6[0].set_ylabel("Length of FSR (sample points)")
ax6[0].plot(FSRs_ix_x, FSRs_ix,'o')
ax6[1].set_title("FSR graph from ITLA, sweep 2")
ax6[1].set_xlabel("FSR location (sample points)")   
ax6[1].set_ylabel("Length of FSR (sample points)")
ax6[1].plot(FSRs_ix_x2, FSRs_ix2,'o')
#frequ_between_peaks = np.sum(FSRs)
#actual_frequ_between_peaks = x_HCN_peaks[first_peak_pd] - x_HCN_peaks[second_peak_pd]
#%% Plotting the ITLA sweeps
#ax4[1].plot(x_sweep1_peaks, cavity_calibration_normalized[x_sweep1_peaks.index],"o", markersize = 2.5)
#ax4[1].plot(x_sweep2_peaks, cavity_calibration_normalized[x_sweep2_peaks.index],'o', markersize = 1)
fig1, ax1 = plt.subplots(2,1, sharex = True)
#ax1[0].plot(HCN_fit_x, fitted_y_HCN) #Obtained from fit block
#ax1[0].plot(range(int(int(result1.params['center']) - result1.params['gamma']/2),int(int(result1.params['center']) + result1.params['gamma']/2)), [result1.params['amplitude']/2 + result1.params['offset']]*int(result1.params['gamma']))
ax1[0].set_title("Sweep 1")
ax1[1].plot(x1, sweep1_cavity,"-o", markersize = 0.5)
ax1[0].plot(x1, sweep1_HCN)
ax1[0].plot(ITLA_first_peak, sweep1_HCN.iloc[ITLA_first_peak],'o')
ax1[1].plot(sweep1_cavity_peaks, sweep1_cavity.iloc[sweep1_cavity_peaks],'o', markersize = 3)
ax1[1].plot(full_sweep_cavity_peaks_1, sweep1_cavity.iloc[full_sweep_cavity_peaks_1],'o', markersize = 1)
ax1[1].plot(sweep1_EOM_ix, sweep1_cavity.iloc[sweep1_EOM_ix],'o', markersize = 1)
ax1[1].plot(indexinit_sweep1,sweep1_cavity.iloc[indexinit_sweep1],'o')
#ax1[1].plot(sweep1_cavity_peaks[init_peak:], sweep1_cavity.iloc[sweep1_cavity_peaks[init_peak:]], 'o', markersize = 1) #Unecessary line, causes some problems.
fig2, ax2 = plt.subplots(2,1, sharex = True)
ax2[0].set_title("Sweep 2")
ax2[1].plot(x2, sweep2_cavity,"-o", markersize = 0.5)
ax2[0].plot(x2, sweep2_HCN)
ax2[1].plot(sweep2_cavity_peaks, sweep2_cavity.iloc[sweep2_cavity_peaks],'o', markersize = 3)
ax2[1].plot(full_sweep_cavity_peaks_2, sweep2_cavity.iloc[full_sweep_cavity_peaks_2],'o', markersize = 1)
ax2[0].plot(ITLA_second_peak, sweep2_HCN.iloc[ITLA_second_peak],'o')
ax2[1].plot(sweep2_EOM_ix, sweep2_cavity.iloc[sweep2_EOM_ix],'o', markersize = 2)
ax2[1].plot(indexinit_sweep2,sweep2_cavity.iloc[indexinit_sweep2],'o')
#ax2[1].plot(sweep2_cavity_peaks[:final_peak], sweep2_cavity.iloc[sweep2_cavity_peaks[:final_peak]], 'o', markersize = 1)
#%%
#Fitting the dispersion curve of laser
fig5, ax5 = plt.subplots(3,1, sharex = "col")
#This is the accurate position of the laser.
ITLA_dispersion_y = np.concatenate((FSRs[0:len(x_sweep1_peaks)], FSRs[len(FSRs) - len(x_sweep2_peaks):]))
ITLA_dispersion_x = np.concatenate((x_sweep1_peaks, x_sweep2_peaks))
#This is a separate version of the dataset to be moved around.
sweep2_dispersion_y = FSRs[len(FSRs) - len(x_sweep2_peaks):]
sweep2_dispersion_x = x_sweep2_peaks
#Try using lmfit
def poly2(x,p):
    return p['a']*(x)**2 + p['b']*x + p['c']
def fcn2min(params,x,y):
    a = params['a']
    b = params['b']
    c = params['c']
    return a*x**2 + b*x + c - y
def f(y,x,p):
    return p['a']*(x)**2 + p['b']*x + p['c'] - y
params = Parameters()
params.add('a', value = 1)
params.add('b', value = 1)
params.add('c', value = 1)
minner = Minimizer(fcn2min,params, fcn_args = (ITLA_dispersion_x,ITLA_dispersion_y))
result = minner.minimize()
report_fit(result)
report = fit_report(result)
fitted_y = poly2(dispersion_x,result.params) 
residual_y = np.abs(poly2(ITLA_dispersion_x, result.params) - ITLA_dispersion_y)
ax5[0].set_title("Dispersion curve from calibration, only range of ITLA")
ax5[0].set_ylabel("FSR (Hz)")
#ax5[0].set_xlabel("Frequency (Hz)")
fig5.text(0.5, 0.04, 'Frequency (Hz)', ha='center')
fig5.text(0.04, 0.5, 'FSR (Hz)', va='center', rotation='vertical')
ax5[1].set_title("Dispersion curve from calibration")
ax5[0].plot(dispersion_x, fitted_y)
ax5[0].plot(ITLA_dispersion_x, ITLA_dispersion_y, 'o', markersize = 2)
ax5[1].plot(dispersion_x, FSRs,'o', markersize = 1)
ax5[2].plot(ITLA_dispersion_x, residual_y, 'o', markersize = 2)
ax5[2].set_title("Frequency difference between fit and data")
ax5[2].set_ylabel("Residual (Hz)")
x = np.flip(np.arange(dispersion_x.iloc[0]- 1e15,dispersion_x.iloc[0], 1e13))
x = dispersion_x
#%% Find the resonance peaks closest to bandwidth

#%% Square of residuals analysis. 
import time
def checkFit(x,y, plot):
    fit_x = np.concatenate((x_sweep1_peaks, x))
    fit_y = np.concatenate((FSRs[0:len(x_sweep1_peaks)], y))
    params = Parameters()
    params.add('a', value = 1)
    params.add('b', value = 1)
    params.add('c', value = 1)
    minner = Minimizer(fcn2min,params, fcn_args = (fit_x,fit_y))
    result = minner.minimize()
    report = fit_report(result)
    fitted_y = poly2(fit_x,result.params)
    residuals= np.abs(fitted_y - fit_y)**2
    sum_residuals = np.sum(residuals)
    if plot:
        fig8, ax8 = plt.subplots()
        ax8.plot(fit_x, fit_y, 'o', markersize = 1)
        ax8.set_xlim(192600000000000, 192740000000000)
        ax8.set_ylim(117690250,117690650)
        ax8.set_title("Optimisation of dispersion curve x-position")
        ax8.set_xlabel("Frequency")
        ax8.set_ylabel("FSR length")
        ax8.plot(dispersion_x, poly2(dispersion_x, result.params))
        fig8.savefig("./FitTest/" + str(int(time.time()*1000)) + ".png")
        plt.close(fig8)
    return sum_residuals

index = x_cavity_peaks.index
FSR_range = int(num_of_FSRs) #Optimal point should be right in the middle
n = 0
xs = []
ys = []
sweep2_no_of_peaks = len(x_sweep2_peaks)
#sweep2_init_peak at 1.92633e+14 Hz
FSR_overlap = 20
sweep2_init_FSR_ix = len(FSRs) - sweep2_no_of_peaks - 1
sweep2_FSRs = FSRs[sweep2_init_FSR_ix:]
sweep2_distances_from_top = [sweep2_FSRs[0]] #Need to forego the HCN finding for sweep 2.

start = len(x_sweep1_peaks) - FSR_overlap
FSR_range = 500 
m = 0
initial_peak_sweep1 = x_sweep1_peaks.index[0] #Sweep_peaks1 and sweep_peaks2 have pandas index, as opposed to sweep1_cavity_peaks etc... Remember they sweep from high to low frequ.
test = x_cavity_peaks[initial_peak_sweep1]
ix = np.where(x_cavity_peaks == test)[0][0]
cavity_init_peak_index = len(x_cavity_peaks[:initial_peak_sweep1])
#This block is optional. It assumes you know the position of the HCN peak.
while m < len(sweep2_FSRs) - 1:
    sweep2_distances_from_top.append(sweep2_distances_from_top[m] + sweep2_FSRs [m+1])
    m += 1
sweeps = []

while n < FSR_range:
    plot = False
    if n == 0:
        x = x_cavity_peaks[initial_peak_sweep1] #This sets the top of the fitting curve. Minus a few indices for wiggle room.
    else: 
        x = x_cavity_peaks[x_cavity_peaks.index[ix + n]] #Plus/minus n to go in opposite direction. Step downwards on dispersion graph with plus.

    chng_x_sweep2_peaks = np.subtract(x, sweep2_distances_from_top) #Make sure that you do not require that.
    if n % 10 == 0:
        plot = False
    residual = checkFit(chng_x_sweep2_peaks,sweep2_FSRs, plot)
    sweeps.append(chng_x_sweep2_peaks)
    xs.append(x)
    ys.append(residual)
    n+=1
    #y = checkFit() #Finish this tomorrow.
fig7, ax7 = plt.subplots()
ax7.plot(xs,ys,"o", markersize = 1)
ax7.set_title("Residuals of fit based on sweep2 position")
ax7.set_ylabel("Sum of residuals")
ax7.set_xlabel("Position of start of sweep 2 (Hz)")
#%%
#Make a function that checks if the residuals function has a minimum.
#If residuals not good enough, maybe use chi squared?
#%% Determine minimum fit error spot
#Each FSR should have its index
ys = np.array(ys)
stitch_frequ_index = np.where(ys == ys.min())[0][0]
stitch_frequ = xs[stitch_frequ_index]
print("Proper start frequ: "  + str(x_sweep2_peaks.iloc[0]))
print("Fitted start frequ: " + str(stitch_frequ))
print("Absolute error: " + str(np.abs(x_sweep2_peaks.iloc[0] - stitch_frequ)))
print("Approx error in FSRs: ", np.abs(x_sweep2_peaks.iloc[0] - stitch_frequ)/117690408.40625)
print("Error in FSRs according to code: ", np.abs(stitch_frequ_index - sweep2_init_FSR_ix)) #Need to fix this part. Might be an overestimation.
ax7.plot(xs[stitch_frequ_index], ys[stitch_frequ_index],'o', label = "Fitted start of sweep 2")
ax7.plot(xs[sweep2_init_FSR_ix], ys[sweep2_init_FSR_ix],'o', label = "Known start of sweep 2")
ax7.legend()
#print("Proper start frequ: ", xs[sweep2_init_FSR_ix - len(x_sweep1_peaks)]) #Not sure what this was about.
#Check if that is actually the proper start FSR.
#Using the squares actually reduces the error to zero.