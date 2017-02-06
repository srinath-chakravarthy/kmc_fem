# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:04:40 2014
Initialize KMC domain from main domain spacing

This subroutine can be split to provide any type of initial distribution
    Initialize t = 0 vacancy distribution
@author: Srinath Chakravarthy (Northeastern University)
"""
import numpy as np


def kmc_solve(lat, N_vac, vac, cx):
    # lat -> lattice
    # vac -> Array of vacancies
    # N_vac -> Total number of vacancies
    for nv in range(N_vac):
        # Check neighbors
        i = vac[nv, 0]  # I position of vacancy in lattice
        j = vac[nv, 1]  # J position of vacancy in lattice
        # Get the rate array
        # ==============================================================================
        #         rate,possible = get_rate_array(lat,i,j)
        #         if (possible):
        #             tot_rate = np.size(rate)
        #             if (tot_rate > 1):
        #                 p2 = np.random.random_integers(0,tot_rate-1)
        #             else:
        #                 p2 = 0
        #             if (p2 < 0 or p2 > tot_rate-1):
        #                 print 'Rate array wrong', nv, p2, tot_rate
        #             else:
        #                 move = rate[p2]
        #                 kmc_particle(lat,vac, N_vac, nv,i,j,move)
        # ==============================================================================
        (lat, vac) = kmc_particle_2(lat, vac, N_vac, nv, i, j)
    # update_conc(lat,vac, cx)
    #        cx = cx/lat.shape[0]

    return (lat, vac)


def update_conc(lat, vac, cx):
    xsize = lat.shape[1]
    for i in range(xsize):
        cx[i] = np.sum(lat[:, i])
        # cx=cx/lat.shape[0]


def get_rate_array(lat, i, j):
    # lat -> is the lattice
    # i -> I position of vacancy in lattice
    # j -> J position of vacancy in lattice
    xsize = lat.shape[0]
    ysize = lat.shape[1]
    l = 0
    # Calculate size of rate array
    if (j + 1 < ysize):
        if (not (lat[i, j + 1])):
            l += 1
    if (j - 1 > -1):
        if (not (lat[i, j - 1])):
            l += 1
    if (i + 1 < xsize):
        if (not (lat[i + 1, j])):
            l += 1
    if (i - 1 > -1):
        if (not (lat[i - 1, j])):
            l += 1
    rate = np.zeros(l, dtype=np.int)
    if (l == 0):
        print('No jumps possible')
        return (rate, False)
    l = 0
    # now fill the rate array
    if (j + 1 < ysize):
        if (not (lat[i, j + 1])):
            rate[l] = 1
            l += 1
    if (j - 1 > -1):
        if (not (lat[i, j - 1])):
            rate[l] = 2
            l += 1
    if (i + 1 < xsize):
        if (not (lat[i + 1, j])):
            rate[l] = 3
            l += 1
    if (i - 1 > -1):
        if (not (lat[i - 1, j])):
            rate[l] = 4
            l += 1
    return (rate, True)


def kmc_particle(lat, vac, N_vac, nv, i, j, move):
    if (move == 1):
        i1 = i
        j1 = j + 1
    if (move == 2):
        i1 = i
        j1 = j - 1
    if (move == 3):
        i1 = i + 1
        j1 = j
    if (move == 4):
        i1 = i - 1
        j1 = j
    lat[i, j] = False
    lat[i1, j1] = True
    vac[nv, 0] = i1
    vac[nv, 1] = j1
    vac[nv, 2] += 1


def kmc_particle_2(lat, vac, N_vac, nv, i, j):
    isize = lat.shape[0]
    jsize = lat.shape[1]
    possible = False;
    k1 = 1
    while (not (possible)):
        p2 = np.random.random_sample()
        if (p2 <= 0.25):
            if (j > 0):
                if (not (lat[i, j - 1])):
                    lat[i, j] = False;
                    lat[i, j - 1] = True;
                    vac[nv, 1] = j - 1;
                    vac[nv, 2] += 1;
                    possible = True;
        elif (p2 > 0.25 and p2 <= 0.5):
            if (j < jsize - 1):
                if (not (lat[i, j + 1])):
                    lat[i, j] = False;
                    lat[i, j + 1] = True;
                    vac[nv, 1] = j + 1;
                    vac[nv, 2] += 1;
                    possible = True;
        elif (p2 > 0.5 and p2 <= 0.75):
            if (i < isize - 1):
                if (not (lat[i + 1, j])):
                    lat[i, j] = False;
                    lat[i + 1, j] = True;
                    vac[nv, 0] = i + 1;
                    vac[nv, 2] += 1;
                    possible = True;
        else:
            if (i > 0):
                if (not (lat[i - 1, j])):
                    lat[i, j] = False;
                    lat[i - 1, j] = True;
                    vac[nv, 0] = i - 1;
                    vac[nv, 2] += 1;
                    possible = True;
        k1 += 1
        if (k1 > 10):
            # istep -= 1
            break
    return (lat, vac)


def remove_vac(lat, vac, nv, N_vac):
    # Remove vacancy number nv from vacancy array
    nv1 = nv + 1
    i = vac[nv, 0]
    j = vac[nv, 1]
    lat[i, j] = False
    if (nv > 0):
        vac1 = np.zeros((nv, 3), dtype=(np.uint16))
        vac2 = np.zeros((N_vac - nv1, 3), dtype=(np.uint16))
        vac1[:, :] = vac[0:nv, :]
        vac2[:, :] = vac[nv1:N_vac, :]
        N_vac -= 1
        vac = np.zeros((N_vac, 3), dtype=(np.uint16))
        vac[0:nv, :] = vac1
        vac[nv:N_vac, :] = vac2
        print('Vacancy ', nv, ' removed')
    else:
        vac2 = np.zeros((N_vac - 1, 3), dtype=(np.uint16))
        vac2[:, :] = vac[1:N_vac, :]
        N_vac -= 1
        vac = np.zeros((N_vac, 3), dtype=(np.uint16))
        vac = vac2
    return (lat, vac, N_vac)


def add_vac(lat, vac, N_vac, i, j):
    # Add a vacancy to some lattice position i,j
    # Since the positions of the vacancy does not matter it is added to the end of the list
    vac2 = np.zeros((N_vac + 1, 3), dtype=(np.uint16))
    vac2[0:N_vac, :] = vac[:, :]
    lat[i, j] = True
    nv = N_vac
    vac2[nv, 0] = i
    vac2[nv, 1] = j
    vac = np.zeros(((N_vac + 1, 3)), dtype=(np.uint16))
    vac = vac2
    N_vac += 1
    return (lat, vac, N_vac)

def add_vacancies(nnatoms, lat, vac, N_vac, ysize):
    for nn in range(nnatoms):
        # ---- Pick a random position on the boundary
        j = 0
        possible = False
        while (not (possible)):
            i = np.random.randint(0, ysize)
            if (not (lat[i, j])):
                possible = True
        # print("Adding vacancy at ", i, j)
        (lat, vac, N_vac) = add_vac(lat, vac, N_vac, i, j)

def remove_vacancies(nnatoms, lat, vac, N_vac, ysize):
    for nn in range(nnatoms):
        # ---- Pick an atom at random on boundary
        # --- Construct an array of vacancies on the boundary and pick one
        N_vac_boundary = np.sum(lat[:, 0])
        if (N_vac_boundary == 0):
            print("Cannot remove atoms")
            break
        if (nn > N_vac_boundary):
            break
        vac1 = np.zeros(N_vac_boundary)
        k = 0
        for i in range(N_vac):
            if (vac[i, 1] == 0):
                vac1[k] = i
                k += 1
        n1 = np.random.randint(0, N_vac_boundary)
        nv = vac1[n1]
        (lat, vac, N_vac) = remove_vac(lat, vac, nv, N_vac)