#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finn Kennedy
12/16/2024
CPU: Apple M1
macOS Sonoma 14.6.1
Spyder, Python 3.11.7
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
import random as rng

rng.seed(123)

nx = 50
ny = 50

spin = np.zeros((nx,ny), np.int8)

N = nx*ny

#initialize spin array (square lattice)
for i in range(0,nx):
    for j in range(0,ny):
        spin[i,j] = rng.choice([-1,1])
        

#initializing the spin array for the hexagonal case
'''
spin = np.zeros((nx,ny,2), np.int8)

#initialize spin array
for i in range(0,nx):
    for j in range(0,ny):
        for k in [0,1]:
            spin[i,j,k] = rng.choice([-1,1])
        
'''
        

#total energy calculation with periodic boundary conditions
#input: spin array, nx, ny are the boundaries, with applied B, 
#J interaction term, and mu = 2*Bohr magneton  
#outputs: Energy of configuration
@numba.jit
def totalEnergy(spin,nx,ny,B,J,mu):
    E = 0
    for i in range(0,nx):
        for j in range(0,ny):
            E += -J*spin[i,j]*(spin[(i+1)%nx,j] + spin[(i-1),j] + spin[i,(j+1)%ny] + spin[i,j-1])
            E += -spin[i,j]*B*mu*2
    return E

#total energy with periodic conditions for Hexagonal Lattices!
#input: spin array(double size for lattice basis), nx, ny are the boundaries, with applied B, 
#J interaction term, and mu = 2*Bohr magneton
#outputs: Energy of configuration
@numba.jit
def totalEnergyHexagonal(spin,nx,ny,B,J,mu):
    E=0
    for i in range(0,nx):
        for j in range(0,ny):
            E += -J*spin[i,j]*(spin[(i+1)%nx,j] + spin[(i-1),j] + spin[i,(j+1)%ny] + spin[i,j-1])
            E += -spin[i,j]*B*mu*2
    return E

#generate a random change in the spin array, Hexagonal case
#input: spin array
#outputs: new spin array with one spin flipped
@numba.jit
def randomChange_Hex(spin):
    spin_new = spin
    x = rng.randrange(0,nx)
    y = rng.randrange(0,ny)
    choice = rng.random()
    if choice > 0.5:
        AorB = 1
    else:
        AorB = 0
    spin_new[x,y,AorB] = -1*spin_new[x,y,AorB]
    return spin_new

#generate a random change in the spin array, 
#input: spin array
#outputs: new spin array with one spin flipped
@numba.jit
def randomChange(spin):
    spin_new = spin
    x = rng.randrange(0,nx)
    y = rng.randrange(0,ny)
    spin_new[x,y] = -1*spin_new[x,y]
    return spin_new

#run the monte-carlo simulation
#input: spin array, functions, number of timesteps,,nx,ny = boundaries, temp = T,B = B field strength
#J interaction term, and two_mu = 2*Bohr magneton
#outputs: array of spin arrays at various timesteps, an array of total energy values at certain timesteps
@numba.jit
def monteCarlo(spin,randomChange,totalEnergy,nx,ny,timesteps,T,J,B,two_mu):
    kb = 8.617e-5 #eV/K
    kbT = kb*T
    E = totalEnergy(spin,nx,ny,B,J,two_mu)
    Eplot = np.zeros(2000)
    n=0
    plot_times = [0,1e3,1e4,1e5,1e6]
    spin_update = []
    for i in range(0,timesteps+1):
        spin_new = randomChange(spin.copy())
        Enew = totalEnergy(spin_new,nx,ny,B,J,two_mu)
        deltaE = Enew - E
        if deltaE <= 0.0:
            spin = spin_new
            E = Enew
        else:
            R = rng.random()
            p = np.exp(-1*(deltaE)/kbT)
            if p > R:
                spin = spin_new
                E = Enew               
        if (i % 500) == 0:
            Eplot[n] = E
            n+=1     
        for j in plot_times:
            if i == j:
                spin_update.append(spin)   
    return spin_update,Eplot

#run the monte-carlo simulation, changing the B field every 100000 timesteps
#input: spin array, functions, number of timesteps,,nx,ny = boundaries, kbT = energy,B = array of B field values
#J interaction term, and two_mu = 2*Bohr magneton
#outputs: magnetization values at every step of B inputted
@numba.jit
def monteCarlo_B_varied(spin,randomChange,totalEnergy,nx,ny,timesteps,kbT,magnetization,B):
    n=0
    E = totalEnergy(spin,nx,ny,B[n],1,2)
    M_plot = np.zeros((300,2))
    for i in range(0,timesteps+1):
        if (i % 10000) == 0:
            M_plot[n,0] = B[n]
            M_plot[n,1] = magnetization(spin,nx,ny)
            n+=1
            E = totalEnergy(spin,nx,ny,B[n],1,2)
        spin_new = randomChange(spin.copy())
        Enew = totalEnergy(spin_new,nx,ny,B[n],1,2)
        deltaE = Enew - E
        if deltaE <= 0.0:
            spin = spin_new
            E = Enew
        else:
            R = rng.random()
            p = np.exp(-1*(deltaE)/kbT)
            if p > R:
                spin = spin_new
                E = Enew               
    return M_plot

#monte-carlo simulation, implemented on a hexagonal lattice
#input: spin array, functions, number of timesteps,,nx,ny = boundaries, temp = T,B = B field strength
#J interaction term, and two_mu = 2*Bohr magneton
#outputs: array of magnetization values at every step of T inputted, an array of total energy values at certain timesteps
@numba.jit
def monteCarlo_Curie_Hex(spin,randomChange,totalEnergy,nx,ny,timesteps,T,magnetization,B,J,two_mu):
    kb = 8.617e-5 #eV/K
    n=0
    l=0
    E_plot = np.zeros(200)
    E = totalEnergyHexagonal(spin,nx,ny,B,J,two_mu)
    M_plot = np.zeros((200,2))
    for i in range(0,timesteps):
        kbT = kb*T[n]
        if (i % 10000) == 0:
            M_plot[n,0] = T[n]
            M_plot[n,1] = (magnetization_hex(spin,nx,ny))**2
            n+=1
        spin_new = randomChange_Hex(spin.copy())
        Enew = totalEnergyHexagonal(spin_new,nx,ny,B,J,two_mu)
        deltaE = Enew - E
        if deltaE <= 0.0:
            spin = spin_new
            E = Enew
        else:
            R = rng.random()
            p = np.exp(-1*(deltaE)/kbT)
            if p > R:
                spin = spin_new
                E = Enew
        if (i % 10000) == 0:
            E_plot[l] = E
            l+=1     
    return M_plot, E_plot

#monte-carlo simulation with changing T every 20,000 timesteps
#input: spin array, functions, number of timesteps,,nx,ny = boundaries, temp = T,B = B field strength
#J interaction term, and two_mu = 2*Bohr magneton
#outputs: array of magnetization values at every step of T inputted, an array of total energy values at certain timesteps
@numba.jit
def monteCarlo_Curie(spin,randomChange,totalEnergy,nx,ny,timesteps,T,magnetization,B,J,two_mu):
    kb = 8.617e-5 #eV/K
    n=0
    l=0
    E_plot = np.zeros(100)
    E = totalEnergy(spin,nx,ny,B,J,two_mu)
    M_plot = np.zeros((100,2))
    for i in range(0,timesteps):
        kbT = kb*T[n]
        if (i % 20000) == 0:
            M_plot[n,0] = T[n]
            M_plot[n,1] = (magnetization(spin,nx,ny))
            n+=1
        spin_new = randomChange(spin.copy())
        Enew = totalEnergy(spin_new,nx,ny,B,J,two_mu)
        deltaE = Enew - E
        if deltaE <= 0.0:
            spin = spin_new
            E = Enew
        else:
            R = rng.random()
            p = np.exp(-1*(deltaE)/kbT)
            if p > R:
                spin = spin_new
                E = Enew
        if (i % 20000) == 0:
            E_plot[l] = E
            l+=1     
    return M_plot, E_plot

#total magnetization of the hexagonal lattice
#returns the sum of all the spins divided by number of spins
@numba.jit
def magnetization_hex(spin,nx,ny):
    N=nx*ny*2
    sum = 0
    for i in range(nx):
        for j in range(ny):
            for k in [0,1]:
                sum += spin[i,j,k]
    return (sum/N)

#total magnetization
#returns the sum of all the spins divided by number of spins
@numba.jit
def magnetization(spin,nx,ny):
    N = nx*ny
    sum = 0
    for i in range(nx):
        for j in range(ny):
            sum += spin[i,j]
    return (sum/N)


#plot figures
'''
#curie temperature curve hexagonal case
#increase T from +200 down to 0.1~0
#record M vs. T
timesteps = 2000000
T = np.linspace(200,.1,200,dtype=np.float32)
J = 0.00144 #eV
two_mu = 0.00011576 # eV/T, = 2*mu_b
B = 5 #T, points out of page in + direction
kb = 8.617e-5 #eV/K
M_plot,E_plot = monteCarlo_Curie_Hex(spin,randomChange_Hex,totalEnergyHexagonal,nx,ny,timesteps,T,magnetization_hex,B,J,two_mu)


plt.figure()
plt.plot(M_plot[:,0],M_plot[:,1])
plt.title ('Magnetization with respect to T')
plt.xlabel('T (K)')
plt.ylabel('M')
plt.show()
'''

#magnetization curve (only did square lattice case for this)
#loop B from 0 to +3, down to -3 and back up to reveal hysteresis
kbT = 1
B1 = np.linspace(0,3,60)
B2 = np.linspace(3,-3,120)
B3 = np.linspace(-3,3,120)
B = np.concatenate((B1, B2, B3))
timesteps = 3000000
M_plot = monteCarlo_B_varied(spin,randomChange,totalEnergy,nx,ny,timesteps,kbT,magnetization,B)

plt.figure()
plt.plot(M_plot[:,0],M_plot[:,1])
plt.title ('Magnetization vs. B field, kbT = ' + str(kbT))
plt.xlabel('B')
plt.ylabel('M')
plt.show()

#plotting real space lattice and spins at each site for square lattice
'''
plot_times = [0,1e3,1e4,1e5,1e6]
Eplot = np.zeros(2000)
B = 5 #T, points out of page in + direction
J = 0.00144 #eV
two_mu = 0.00011576 # eV/T, = 2*mu_b
#T, points out of page in + direction
kb = 8.617e-5 #eV/K
timesteps = 1000000
T = 15
spin_update,Eplot = monteCarlo(spin,Eplot,randomChange,totalEnergy,nx,ny,timesteps,T,J,B,two_mu)

#structure plots
for i in range(len(spin_update)):
    spin = spin_update[i]
    spin_up_x = []
    spin_up_y = []
    spin_down_x = []
    spin_down_y = []
    plt.figure()
    for x in range(0,nx):
        for y in range(0,ny):
            if spin[x,y] == 1: #if spin is up
                spin_up_x.append(x)
                spin_up_y.append(y)
            else:
                spin_down_x.append(x)
                spin_down_y.append(y)
    plt.scatter(spin_down_x,spin_down_y,color = 'blue',s=5,marker='o')
    plt.scatter(spin_up_x,spin_up_y,color = 'red',s=5, marker='o')
    plt.title ('Spins at timestep = ' + str(plot_times[i]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#total energy vs. timestep plots
plt.figure()
plt.plot(Eplot)
plt.title ('Total Energy')
plt.xlabel('# of time steps (in 500s)')
plt.ylabel('U')
plt.show()
'''
#plotting real space lattice and spins at each site for hexagonal lattice
'''
#structure plots
for i in range(len(spin_update)):
    sc = np.sqrt(3)/2
    sep = 1/6
    spin = spin_update[i]
    spin_up_x = []
    spin_up_y = []
    spin_down_x = []
    spin_down_y = []
    plt.figure()
    for x in range(0,nx):
        for y in range(0,ny):
            if (y%2) == 0:
                if spin[x,y,0] == 1: #if spin is up
                    spin_up_x.append(x-sep)
                    spin_up_y.append(y*sc)
                if spin[x,y,1] == 1: #if spin is up
                    spin_up_x.append(x+sep)
                    spin_up_y.append(y*sc)
                if spin[x,y,0] == -1:
                    spin_down_x.append(x-sep)
                    spin_down_y.append(y*sc)
                if spin[x,y,1] == -1:
                    spin_down_x.append(x+sep)
                    spin_down_y.append(y*sc)
            else:
                if spin[x,y,0] == 1: #if spin is up
                    spin_up_x.append(x-sep+0.5)
                    spin_up_y.append(y*sc)
                if spin[x,y,1] == 1: #if spin is up
                    spin_up_x.append(x+sep+0.5)
                    spin_up_y.append(y*sc)
                if spin[x,y,0] == -1:
                    spin_down_x.append(x-sep+0.5)
                    spin_down_y.append(y*sc)
                if spin[x,y,1] == -1:
                    spin_down_x.append(x+sep+0.5)
                    spin_down_y.append(y*sc)
    plt.scatter(spin_down_x,spin_down_y,color = 'blue',s=10,marker='o')
    plt.scatter(spin_up_x,spin_up_y,color = 'red',s=10, marker='o')
    plt.title ('Spins at timestep = ' + str(plot_times[i]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#total energy vs. timestep plots
plt.figure()
plt.plot(Eplot)
plt.title ('Total Energy')
plt.xlabel('# of time steps (in 500s)')
plt.ylabel('U')
plt.show()
'''
#magnetization calculation wrt. T for square lattice
'''
#curie temperature curve
#increase T from zero
#record M vs. T
timesteps = 2000000
T = np.linspace(90,.1,100,dtype=np.float32)
J = 0.00144/3 #eV
two_mu = 0.00011576 # eV/T, = 2*mu_b
B = 0.5 #T, points out of page in + direction
kb = 8.617e-5 #eV/K
M_plot,E_plot = monteCarlo_Curie(spin,randomChange,totalEnergy,nx,ny,timesteps,T,magnetization,B,J,two_mu)

plt.figure()
plt.plot(M_plot[:,0],M_plot[:,1])
plt.title ('Magnetization with respect to T')
plt.xlabel('T (K)')
plt.ylabel('M')
plt.show()
'''

#edit this based on what plot you want to export.
output_filename = "name.csv"
np.savetxt(output_filename, avg_array, delimiter=",")
print(f"Data exported to {output_filename}")
