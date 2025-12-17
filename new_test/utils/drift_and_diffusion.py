
import numpy as np
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
    'axes.labelsize': 18,'axes.titlesize': 28, 'figure.titlesize' : 28})
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pandas as pd
from kramersmoyal import km
import scipy as sc
import seaborn as sns
import sys
import os
from scipy.stats import skew, kurtosis
sys.path.append('../')
from new_test.utils.dataloading import *
from new_test.utils.statistical_tests import *

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, ProcessPoolExecutor

def fit_km_1_linear(edges, km_1):
    mid_point = np.argmin(edges**2)
    start = mid_point - 500
    end = mid_point + 500
    return sc.optimize.curve_fit(lambda t,a,b: a-t*b, edges[start:end], km_1[start:end])[0]

def fit_km_1_nonlinear_params(x, edges, km_1):#, edges, km_1):
    start = np.argmax(edges > -0.22)
    end = np.argmax(edges > 0.22)-1
    edges_pol = edges[start:end]
    pol = np.polyfit(edges_pol, km_1[start:end], 3)
    p3, p2, p1, p0 = pol
    return p3, p2, p1, p0


def fit_km_2_nonlinear_params(x, edges, km_2): #, edges, km_2):
    start = np.argmax(edges > -0.22) #-0.22, 0.22
    end = np.argmax(edges > 0.22)-1
    edges_pol = edges[start:end]
    pol = np.polyfit(edges_pol, (km_2[start:end]), 4) # fit the 2nd KM coefficient directly
    p4, p3, p2, p1, p0 = pol
    return p4, p3, p2, p1, p0

def Euler_Maruyama(p0, p1, p2, p3, q0, q1, q2, q3, q4, c_1 = 0.1, epsilon_fixed = 0.1,delta_t = 0.01, number_of_days = 111):
    t_final = 3600*24*number_of_days # Anzahl der Tage
    time = np.arange(0.0, t_final, delta_t)
    theta = np.zeros(time.size)
    omega = np.zeros(time.size)
    # set a seed for reproducibility
    np.random.seed(42)
    dW = np.random.normal(0, np.sqrt(delta_t), time.size)
    omega[0] = 0
    theta[0] = 0

    for model in ['linear', 'non-linear']:
        if model == 'linear':
            for i in range(1,time.size):

                '''use the P here that is calculated from the power_mismatch function above !!! '''
                omega[i] = omega[i-1] - c_1*(omega[i-1])*delta_t + epsilon_fixed*dW[i-1]
            omega_linear = omega.copy()
        elif model == 'non-linear':
            dW = np.random.normal(0, np.sqrt(delta_t), time.size) # define different dW for the non-linear model
            omega = np.zeros(time.size)
            omega[0] = 0
            for i in range(1,time.size):

                omega[i] = omega[i-1] + (p3*(omega[i-1])**3 + p1*omega[i-1] + p0)*delta_t + np.sqrt(2*(q4*omega[i-1]**4 + q3*omega[i-1]**3 + q2*omega[i-1]**2 + q1*omega[i-1] + q0))*dW[i-1] 
            omega_nonlinear = omega.copy()
            return omega_linear, omega_nonlinear
