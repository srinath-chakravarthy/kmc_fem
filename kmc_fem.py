"""
Basic kmc class
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:51:36 2014
Created on Mon Apr 21 09:19:04 2014

Kinetic Monte Carlo for Bulk Diffusion with coupled Finite elements

D = (1/4)Γ(a^2)

@author: Srinath Chakravarthy (Northeastern University)
"""

import numpy as np
import math
import matplotlib
#matplotlib.use('Cairo')
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sp
from scipy.sparse import *
from fe import *
from kmc import *
import time

rc('text', usetex=True)
#rc('font', family='serif')
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.labelspacing'] = 0.25
plt.rcParams['legend.borderpad'] = 0.0
plt.rcParams['legend.borderaxespad'] = 0.1
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.major.size']=6
plt.rcParams['ytick.major.size']=6
plt.rcParams['xtick.labelsize']='large'
plt.rcParams['ytick.labelsize']='large'
mark = ['k-','r-','b-','m-', 'g-','y-','k--','r--','b--','g--','m--'];
#%%---------------------------------------------------------------------------------------
#---------------Input Parameters ----------------------------
a_lat = 2.5e-10                     # Lattice spacing (m)
D = 5e-9                            # Diffusion Coefficient (m^2/s)
ysize = np.int(1001)                 # Number of sites in Y direction
xsize = np.int(201)                 # Number of sites in X direction
hop_rate = 4.0*D/(a_lat**2);        # Hopping rate for square lattice
dt = 1.0/hop_rate;                  # Default average time for a hop of a single particle
beta = 0.5                          # Default Numerical Time integration constant
theta = 0.5                         # Default weighted factor for iterative KMC FE convergence
# This is assumed to be the average hop time for N particles
#%%---------------------------------------------------------------------------------------
# ----------------- Timers ----------------------
d_tau = dt                          # time increment for motion of N particles
tau_update = 20                     # Update of KMC and FEM
# d_tau_update is the FEM time step
T_max = 400                         # Total time increments for time domain
DT = 60                             # Non-dimensional time step for time increment
Tau_max = DT*T_max                 # Total non-dimensional KMC time
n_sub_increments= DT/tau_update     # Number of subincrements within a time increment
dt_fem = n_sub_increments/hop_rate
nsteps=11

# Other parameters controlling the KMC simulation TBD
    # Width of band for KMC
    # Length over which averaging of KMC concentrations
#nsteps = tau_max                    # Total number of FE/KMC steps
nplot = 2                         # Fraction of total FE/KMC steps to plot data
nplotstep = int(nsteps/nplot)            # Number of total FE/KMC steps to plot data
#%% ------------------------------------------------------------------------------------------
# -------------------- KMC/FE domain split ---------------------------------------------------------

#==============================================================================
kmcxsize = np.int(xsize/2 +1)               # Size of KMC region
femxsize = xsize - kmcxsize + 1             # Size of FEM region
tot_xsize = np.int(kmcxsize + femxsize-1)     # Total size of the region
#==============================================================================
#femxsize = xsize
x_kmc = np.zeros(kmcxsize)
x_fem = np.zeros(femxsize)
x = np.zeros(tot_xsize)
# ----------- Initialize fem and KMC domains
for i in range(femxsize):
    x_fem[i]=i*a_lat;
for i in range(kmcxsize):
    x_kmc[i]=x_fem[femxsize-1]+i*a_lat
x[0:femxsize]=x_fem
x[femxsize:tot_xsize]=x_kmc[1:kmcxsize]
##x_kmc = np.zeros(kmcxsize)
##l1 = femxsize -1
##for i in ()
#%% ------------------------------------------------------------------------------------------
# ------------------ Initialize FEM and KMC and apply Boundary conditions ---------------------
conc_x = np.zeros(tot_xsize)            # Main Concentration array, currently setup only for 1D problems
## Array Organization for the concentration
    # conc_x(0:femxsize-1) --> Fem Field solution
    # conc_x(femxsize,xsize) --> KMC solution
## Initial Condition
    # conc_x(x,t=0) --> Guassian distribution
sigma_conc = 30.0                   # Standard Deviation of Gaussian Distribution
act_dist_fem = np.zeros_like(x_fem);
act_dist_kmc = np.zeros_like(x_kmc);
mu = x_fem[femxsize-1]/a_lat;
mu = 0.0

for i in range(femxsize):
    act_dist_fem[i] = 1/(sigma_conc * np.sqrt(2*np.pi))*np.exp(-(x_fem[i]/a_lat - mu)**2 / (2 * sigma_conc**2) )

for i in range(kmcxsize):
    act_dist_kmc[i] = 1/(sigma_conc * np.sqrt(2*np.pi))*np.exp(-(x_kmc[i]/a_lat - mu)**2 / (2 * sigma_conc**2) )

act_dist_fem = act_dist_fem*10.0;
act_dist_kmc = act_dist_kmc*10.0;
conc_x[0:femxsize]=act_dist_fem
conc_x[femxsize:tot_xsize]=act_dist_kmc[1:kmcxsize]
conc_kmc = np.zeros(kmcxsize)
conc_kmc = act_dist_kmc

#%% ------------------------------------------------------------------------------------------
#------------- Initialize FEM ---------------------------
(K, M, Kp, Kpp, RHS)= fe_init(D,femxsize,x_fem,dt_fem, beta)
    # ---- Store row sums of Kp ---
pf = np.zeros(femxsize)
for i in range(femxsize):
    pf[i] = Kp.getrow(i).sum()
    # ---- Apply Boundary conditions
RHS_new = np.zeros_like(RHS)
#------------- Initialize KMC ---------------------------
lat = np.zeros((ysize,kmcxsize),dtype=np.bool)
N_vac = np.int(0);
# Generate vacancies along every column according to guassian
for j in range(kmcxsize):
    p1 = np.random.rand(ysize)-0.5/ysize
    lat[p1<=act_dist_kmc[j],j]=True;
N_vac = np.sum(lat);

vac=np.zeros((N_vac,3), dtype=(np.uint16));
k = np.int(0)

for i in range(ysize):
    for j in range(kmcxsize):
        if (lat[i,j]):
            vac[k,0]=i;
            vac[k,1]=j;
            k +=1
update_conc(lat,vac,conc_kmc)
conc_kmc = conc_kmc/lat.shape[0]
#==============================================================================
# plt.figure()
# plt.plot(x_fem/a_lat,conc_x[0:femxsize])
# plt.plot(x_kmc/a_lat,conc_kmc)
# plt.show()

#==============================================================================

#%% ------------------------------------------------------------------------------------------
# ----------- Loop through Solution steps -----------------------
conc_fem_x = conc_x[0:femxsize]
t1 = time.clock()
kplot = 0                                               # Plotting increment
Tincr = 0                                               # Time increment in terms of d_T
istep= 0                                                # KMC step counter

vac_orig = np.zeros_like(vac)
vac_orig = vac

fem_boundary = femxsize-1
kmc_boundary = 0
print ("Initial Flux at boundary = ", (K*conc_fem_x)[fem_boundary])
lambda1 = np.zeros(int(n_sub_increments))
tstep = 0
for istep in range(nsteps):                                 # Main Time increment loop, solution proceeds in DT increments
    conc_x1 = conc_fem_x
    converged=False
    iter1 = 0
    c2 = np.zeros(int(n_sub_increments))
    lambda1[0:] = conc_kmc[kmc_boundary]
    while(not converged):
        eps1 = 0.0
        # Fem time scale is essentially the n_sub_increments
        # Always do 1st iteration
        RHS[fem_boundary] = -K.dot(conc_x1)[fem_boundary]
        for inc in range(int(n_sub_increments)):
            for itau in range(tau_update):                   # KMC loop over the update time scale
                (lat,vac) = kmc_solve(lat, N_vac, vac, conc_kmc)
                c1 = np.sum(lat[:,kmc_boundary])/np.float(ysize) # concentration at the end of tau_update steps
                conc_kmc[kmc_boundary]=c1
                if (c1 == 0):
                    c1p = 1.0/np.float(ysize)
                else:
                    c1p = c1
            natoms= round((lambda1[inc] - conc_kmc[kmc_boundary]) * ysize)
            if (np.abs(natoms) > 0):
                if (natoms > 0):
                    # Add atoms
                    #print("Adding vacancies ", istep, iter1, natoms)
                    add_vacancies(np.int(np.abs(natoms)),lat,vac,N_vac,ysize)
                else:
                    # Subtract atoms
                    #print("Removing vacancies ", istep, iter1, natoms)
                    remove_vacancies(np.int(np.abs(natoms)), lat, vac, N_vac, ysize)
            RHS_new[fem_boundary] = natoms / ysize * a_lat / 2.0 / dt_fem  # Stores the flux at each of the update steps
            conc_fem = Kp.dot(Kpp.dot(conc_fem_x) + (1 - beta) * RHS + beta * RHS_new)  # conc at time T+inc*deltat
            RHS = RHS_new
            c2[inc] = conc_fem[fem_boundary]
            eps1 += (1.0/DT)*(c1-c2[inc])**2*ysize/c1p*(1.0*tau_update) # here 1.0 is there because each kmc step is exactly 1.0
        eps2 = np.sqrt(eps1)
        if (eps2 <= 1.0):
            converged = True
            conc_fem_x = conc_fem
            print ("Step ", istep, "converged in ", iter1, "iterations", eps1)
            eps1 = 0.0
        else:
            iter1 += 1
            eps1 = 0.0
            print ("Step = ", istep, "iteration = ", iter1, " Error = ", eps2)
            # Update lambda
            for inc in range(int(n_sub_increments)):
                lambda1[inc] = theta*c2[inc] + (1.0 - theta)*lambda1[inc]
            if (iter1 > 30):
                converged = True
    if (istep%nplotstep == 0):
        print("Plotting Step ", istep)
        for i in range(kmcxsize):
            conc_kmc[i]=np.sum(lat[:,i])
        conc_kmc = conc_kmc/lat.shape[0]
        filename="kmc_coupled_fe_" + str(istep) + ".dat"
        plt.plot(x_fem/a_lat, conc_fem_x, mark[kplot])
        plt.plot(x_kmc[1:kmcxsize]/a_lat, conc_kmc[1:kmcxsize], mark[kplot])
        #plt.show()

        write1 = np.zeros((tot_xsize,2))
        write1[0:femxsize,0]=x_fem/a_lat
        write1[femxsize:,0]=x_kmc[1:kmcxsize]/a_lat
        write1[0:femxsize,1]=conc_fem_x
        write1[femxsize:,1]=conc_kmc[1:kmcxsize]

        np.savetxt(filename,write1)
        #plt.plot(write1[:,0],write1[:,1], mark[kplot])

        kplot += 1
    istep += 1

#plt.legend(loc=0)
plt.xlabel(r'$\zeta = x/a$',fontsize=20)
plt.ylabel(r'Concentration, $c(\zeta)$',fontsize=20)
plt.show()