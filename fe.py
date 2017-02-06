# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:06:41 2014

Initialize 1D Finite Element mesh for coupling with KMC
The full domain is split between KMC and FE

Currently the FE mesh is taken in from the main program
  This will be done by some preprocessor later
Steps
1) Form 1D mesh
2) Assemble 1D mesh
3) Apply Boundary conditions
4) Apply initial conditions
5) Find the inverse of the Matrix and store it.

Inputs
  a) Diffusion coefficient (Material Properties)
  b) Number of nodes and their positions
  c) Hop_rate
  d) Time step (We will assume unconditionally stable) with Newmark integration
  Element matrix in this case is trivial for linear 1D elements
      [K] = D/L |1 -1|    [M]= L/2 |1 0|   or |2 1|
                |-1 1|             |0 1|      |1 2|
@author: Srinath Chakravarthy (Northeastern University)
"""
import numpy as np
import scipy as sp
from scipy.sparse import *
from scipy.sparse.linalg import *


def fe_init(D, nnodes, x_n, dT, beta):
    # D --> Diffusion coefficient (per unit area ???)
    # nnodes --> Number of nodes
    # x_n --> Position of nodes (size is nnodes)
    # hop_rate --> Hopping rate for discrete problem
    # dT --> Actual dT for the problem ()
    # Empty sparse matrices for
    # K ---> Stiffness matrix
    # M ---> Mass matrix
    nelem = nnodes - 1  # Number of Element (nnodes -1 --> 1D)
    icon = np.zeros((nelem, 2), dtype=np.int)  # Nodal Connectivity (nelem, 2 --> 1D)
    # 2 Nodes per element
    ele_len = np.zeros(nelem)
    # %% --------------------------------------------------------------------------------
    # ---------- Calculate Element Connectivity and Lengths
    for i in range(nelem):
        icon[i, 0] = i
        icon[i, 1] = i + 1
        ele_len[i] = np.abs(x_n[i + 1] - x_n[i])
    # %% --------------------------------------------------------------------------------
    # ----- Assembly of Global Stiffness matrix -----------------------------------------
    K = dok_matrix((nnodes, nnodes))
    M = dok_matrix((nnodes, nnodes))
    Kp = dok_matrix((nnodes, nnodes))
    Kpp = dok_matrix((nnodes, nnodes))
    RHS = np.zeros(nnodes)
    for i in range(nelem):
        coeffK = D / ele_len[i]
        coeffM = ele_len[i] / 2.0
        for j in range(2):
            l1 = icon[i, j]
            for k in range(2):
                l2 = icon[i, k]
                if (l1 == l2):
                    K[l1, l2] += coeffK
                    M[l1, l2] += coeffM
                else:
                    K[l1, l2] -= coeffK
                    M[l1, l2] += 0
    Kp = M / dT + beta * K
    Kpp = M / dT - (1.0 - beta) * K
    Kp1 = Kp.tocsc()
    Kp1 = sp.sparse.linalg.inv(Kp1)
    return (K, M, Kp1, Kpp, RHS)


def fe_solve(Kpp, RHSti, RHSti1, pf, cx, beta):
    # In this simple case RHS = 0 at all times so it does not play a role
    cx1 = np.zeros_like(cx)
    cx1 = cx
    cx = Kpp.dot(cx1) + (1.0 - beta) * RHSti + beta * RHSti1
    for i in range(np.size(cx)):
        cx[i] = cx[i] / pf[i]
    return (Kpp, cx)


