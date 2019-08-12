#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


# import numpy as np
from numpy import *
import numpy as np

from scipy import optimize
from scipy.integrate import solve_ivp, cumtrapz
from scipy.linalg import null_space, expm, inv, norm, eig

# from matplotlib.pyplot import *
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# from matplotlib.animation import FuncAnimation

from types import SimpleNamespace

import inspect


# ### Geometry variables

# In[ ]:


# Todo: Define rho then get n_cables and n_dims from rho matrix
n_cables = 1
n_dims = 3

# TODO: Different cable lengths for different cables
cable_length = 1.23

# Array of cable lengths
lq = cable_length * ones(n_cables).reshape(-1,1,1)

# Load - Rigid body dimensions
length = .05
width = .05
height = .05

# Matrix of attachment points in body frame
# rho = array([[[length/2],  [width/2], [-height/2]],
#                [[-length/2], [ width/2], [-height/2]],
#                [[-length/2], [-width/2], [-height/2]],
#                [[ length/2], [-width/2], [-height/2]]])

# Matrix of attachment points in body frame
rho = array([ [[0], [0], [-height/2]] ])


# ## Control gains and simulation parameters

# In[ ]:


# Integration times

t0 = 0
tf = 10

# DEBUG selectors

# For a complete Lyapunov function with negative semi-definite time derivative 
NO_U_PERP_CROSS_TERMS = False

# TLd derivatives are zero (emulate solution of optimization problem)
TLD_DERIVATIVES_ZERO = False


# ### Control gains

# In[ ]:


L_q = 50
L_oq = 10

L_qR = 100
L_oqR = 10

kp = 1.0
kv = 1.0

x_pv = (kp + kv) / 2

kq = 20
koq = 5

kqR = 100
kqo = 10

# DEBUG
TL_repulsive_gain = 0.0
TL_damping = 0.0

mL = 0.3
mass_quad = 1.280
mQ = mass_quad * ones([n_cables,1,1])

g = 9.8

# inertia matrix for the quadrotors
J = diag([0.21, 0.22, 0.23])
inv_J = inv(J)

J = stack(n_cables*[J])
inv_J = stack(n_cables*[inv_J])

e1 = array([[1,0,0]]).T
e2 = array([[0,1,0]]).T
e3 = array([[0,0,1]]).T

# Minimum angle with vertical of z-body direction when optimizing
theta_min_for_optimization = deg2rad(20)


# ## Load trajectory

# In[ ]:


import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.autowrap import ufuncify

t = sp.symbols('t', real=True)

r,rx,ry = sp.symbols('r r_x r_y')
omega,omega_x,omega_y = sp.symbols('omega omega_x omega_y')
theta_x,theta_y = sp.symbols('theta_x theta_y')
alpha = sp.symbols('alpha')

### Trajectory parameters

# # Circle
# args = {rx:10,
#        ry:10,
#        omega_x: 1,
#        omega_y: 1,
#        theta_x: sp.pi/2,
#        theta_y: 0}

# # Oval
# args = {rx:10,
#        ry:15,
#        omega_x: 1,
#        omega_y: sp.S(1)/2,
#        theta_x: sp.pi/2,
#        theta_y: 0}

# Lemniscate
args = {rx:10,
       ry:15,
       omega_x: 1,
       omega_y: sp.S(1)/2,
       theta_x: 0,
       theta_y: 0}

# Lemniscate
args = {rx:1,
       ry:2,
       omega_x: sp.S(1)/2,
       omega_y: (sp.S(1)/2)/2,
       theta_x: 0,
       theta_y: 0,  
       theta_x: sp.pi/4,
       theta_y: sp.pi/4,  
#        omega_x: 1e-3,
#        omega_y: 1e-3,
#        theta_x: sp.pi/2+1e-3,
#        theta_y: sp.pi/2+1e-3,
       }


# Trajectory definition
rx = 0
ry = 0
omega_x = 0
omega_y = 0
p_symb = sp.Matrix([rx*sp.sin(omega_x*t+theta_x),
                    ry*sp.sin(omega_y*t+theta_y),
                     -1.50 + 0*t])


# In[ ]:


p_vec = stack([ufuncify(t,p_symb[i].subs(args)) for i in arange(n_dims)])
v_vec = stack([ufuncify(t,p_symb.diff(t)[i].subs(args)) for i in arange(n_dims)])
a_vec = stack([ufuncify(t,p_symb.diff(t,2)[i].subs(args)) for i in arange(n_dims)])
da_vec = stack([ufuncify(t,p_symb.diff(t,3)[i].subs(args)) for i in arange(n_dims)])
d2a_vec = stack([ufuncify(t,p_symb.diff(t,4)[i].subs(args)) for i in arange(n_dims)])

def pr(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([p_vec[i](x) for i in range(len(p_vec))],axis=1).reshape(-1,1,3,1)

    # If first dimension is singleton then remove it
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec

def dpr(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([v_vec[i](x) for i in range(len(v_vec))],axis=1).reshape(-1,1,3,1)

    # If first dimension is singleton then remove it
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec

def d2pr(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([a_vec[i](x) for i in range(len(a_vec))],axis=1).reshape(-1,1,3,1)

    # If first dimension is singleton then remove it
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec

def d3pr(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([da_vec[i](x) for i in range(len(a_vec))],axis=1).reshape(-1,1,3,1)

    # If first dimension is singleton then remove it
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec

def d4pr(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([d2a_vec[i](x) for i in range(len(a_vec))],axis=1).reshape(-1,1,3,1)

    # If first dimension is singleton then remove it
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec


# ### Auxiliar functions

# In[ ]:


def skew(v):
    # New shape is v shape with last dimension 3 instead of 1 (3x1 vector --> 3x3 matrix)
    s = asarray(v.shape)
    s[-1] = 3
    
    # Vectorize
    # Maintains the shape of v for vi
    v0 = v[...,0,0,newaxis,newaxis]
    v1 = v[...,1,0,newaxis,newaxis]
    v2 = v[...,2,0,newaxis,newaxis]

    z = zeros_like(v0)
    
    # TODO: Check that dim(v) = 3
    res = concatenate([z, -v2, v1,
                    v2, z, -v0,
                    -v1, v0, z],axis=-2)
    
    return res.reshape(s)

def unskew(M):
    # New shape is M shape with last dimension 1 instead of 3 (3x3 matrix --> 3x1 vector)
    s = asarray(M.shape)
    s[-1] = 1

    # TODO: Check if M is anti-symmetric
#     TODO: stack along axis=-1 instead of using reshape()
    
    res = stack([-M[...,1,2].reshape(-1,1),
                  M[...,0,2].reshape(-1,1),
                  -M[...,0,1].reshape(-1,1)],axis=1).reshape(s)
    
    return res

# TODO: Check that unskew(skew(v)) == v
# TODO: Do not make assumptions on shape of input


# In[ ]:


def mt(M):
# Matrix transpose no matter what is M shape.  M = (...) x n x m or M = (...) x m x n
    
    return np.swapaxes(M,-1,-2)


# In[ ]:


# TODO: Vectorize

# Numpy rotation matrices around x,y,z axis

def Rx(theta):
    return array([[1, 0, 0],
                   [ 0, cos(theta), -sin(theta)],
                   [0, sin(theta), cos(theta)]])

def Ry(theta):
    return array([[cos(theta), 0, -sin(theta)],
                   [0., 1, 0],
                  [sin(theta), 0, cos(theta)]])
                   

def Rz(theta):
    return array([[cos(theta), -sin(theta), 0],
                   [sin(theta), cos(theta), 0],
                   [0,0, 1]])


# In[ ]:


# TODO: vectorize

# From: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# Calculates Rotation Matrix given euler angles.
# theta :: [roll pitch yaw]
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,          0,                 0                ],
                    [0,         np.cos(theta[0]),  -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]),   np.cos(theta[0]) ]])

    R_y = np.array([[ np.cos(theta[1]),   0,      np.sin(theta[1])  ],
                    [ 0,                  1,      0                 ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]])

    R_z = np.array([[np.cos(theta[2]),   -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),    0],
                    [0,                   0,                   1]])

    R = R_z @ R_y @ R_x

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
#     return True
#     return amax(abs((np.eye(3) - mt(R)@R))) < 1e-4
    return amax(abs((np.eye(3) - mt(R)@R))) < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = np.sqrt(R[...,0,0] * R[...,0,0] +  R[...,1,0] * R[...,1,0])

    singular = sy < 1e-6

    x = array(np.arctan2(R[...,2,1] , R[...,2,2]))
    y = array(np.arctan2(-R[...,2,0], sy))
    z = array(np.arctan2(R[...,1,0], R[...,0,0]))
    
    
    x[singular] = np.arctan2(-R[singular,1,2], R[singular,1,1])
    y[singular] = np.arctan2(-R[singular,2,0], sy[singular])
    z[singular] = 0

    return np.array([x, y, z])


# In[ ]:


def check_derivative(f,df,t,tol=1e-6):
    int_df = cumtrapz(df,t,axis=0,initial=0)
    max_error = amax(abs(f-int_df - (f[0]-int_df[0])))
    if(max_error > tol):
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        print(f"High derivative error: tol < {max_error:.2E} \t Line {caller.lineno:3}: {caller.code_context}")
    return

def check_equality(arg1,arg2,tol=1e-6):
    max_error = amax(abs(arg1-arg2))
    if(max_error > tol):
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        print(f"High state error: tol < {max_error:.2E} \t Line {caller.lineno:3}: {caller.code_context}")
    return


# ## Problem definition

# ### Contruction of P matrix and its nullspace

# In[ ]:


def matrix_nullspace(R=eye(3)):
    # Computes P matrix and its nullspace basis G for a certain load attitude
    
    # TLd = [TLd_11, TLd_12, .... , TLd_nn-1 , TLd_nn]

    # Sum of cable tensions in each axis
    P = kron( ones(n_cables), eye(n_dims))

#     P = vstack([
#         P,
#         hstack([skew(R @ rho[i]) for i in range(n_cables)])   # Only care about load yaw control (not pitch or roll)
#     ])

    # G is subspace of allowable TLd derivatives (which do not affect F or M)
    G = null_space(P)

    return (P, G)


# ### Load PD control functions

# In[ ]:


# From https://github.com/gabrii/Smoothsteps/blob/master/functions/7.txt

left_edge = 0.2
right_edge = 0.75
    
def smooth(x, left_edge=left_edge, right_edge=right_edge):
    x = (x - left_edge) / (right_edge - left_edge)
    return  -20*x**7 + 70*x**6 - 84*x**5 + 35*x**4

def Dsmooth(x, left_edge=left_edge, right_edge=right_edge):
    x = (x - left_edge) / (right_edge - left_edge)
    slope = 1 / (right_edge - left_edge)
    return  slope*(-20*7*x**6 + 70*6*x**5 - 84*5*x**4 + 35*4*x**3)

def log_barrier(x):
    x = abs(x)
    x = clip(x,left_edge,right_edge)
    x = smooth(x,left_edge,right_edge)
    x = maximum(1e-3,x)
    return -log(x)

def Dlog_barrier(x):
    x_orig = x
    s = sign(x)
    x = abs(x)
    Dx = Dsmooth(x,left_edge,right_edge)
    x = clip(x,left_edge,right_edge)
    x = smooth(x,left_edge,right_edge)
    x = maximum(1e-3,x)
    der = -s*Dx/x * (abs(x_orig) > left_edge) * (abs(x_orig) < right_edge)
    return der


# In[ ]:


def F_feedback(state):
#     import inspect
#     frame = inspect.currentframe()
#     print("state.p @ F_feedback(): ", hex(id(state.p)))
#     try:
#         for key,val in state.__dict__.items(): 
#             frame.f_back.f_locals[key]=val
#     finally:
#         del frame
    
#     print("p @ F_feedback():", hex(id(p)))

    p = state.p
    v = state.v
    pd = state.pd
    dpd = state.dpd
    d2pd = state.d2pd
    
    # PD controller
    F = mL * (- kp*(p-pd) 
             - kv*(v-dpd) 
             - g*e3 
             + d2pd)
    
    try:
        dv = state.dv
        d3pd = state.d3pd
        
        dF = mL * (- kp*(v-dpd) 
                 - kv*(dv-d2pd) 
                 + d3pd)
    except Exception as e: 
#         print(e)
        return F
    
    try:
        d2v = state.d2v
        d4pd = state.d4pd
        
        d2F = mL * (- kp*(dv-d2pd) 
                 - kv*(d2v-d3pd) 
                 + d4pd)
    except Exception as e: 
#         print(e)
        return (F,dF)
    
    return (F,dF,d2F)





# Cable repulsive function
def TL_repulsive(TL):
    
    cable_matrix = (- TL / norm(TL, axis=-2, keepdims=True)) * lq
    
    quad_positions = - cable_matrix + rho

    dTL = - TL_repulsive_gain * Dlog_barrier(quad_positions)
    
    dTL[...,2,0] = 0    # No repulsion along z-axis

    # TODO: Raise error
    if(isnan(dTL).any()):
        print('quad_positions:')
        print(quad_positions)
        print('Dlog_barrier:')
        print(dTL)
    
    return dTL



def cable_tensions(state):
    TLd_min = state.TLd_min
    delta_TLd = state.delta_TLd
 
    # TODO: Compute correct derivatives!
    
    TLd = TLd_min + delta_TLd
    
    try:
        dTLd_min = state.dTLd_min
        ddelta_TLd = state.ddelta_TLd
        
        dTLd = dTLd_min + ddelta_TLd
        if (TLD_DERIVATIVES_ZERO):
            dTLd = 0*dTLd;
    except Exception as e: 
#         print(e)
        return TLd

    try:
        d2TLd_min = state.d2TLd_min
        d2delta_TLd = state.d2delta_TLd
    
        d2TLd = d2TLd_min + d2delta_TLd
        if (TLD_DERIVATIVES_ZERO):
            d2TLd = 0*d2TLd;
    except Exception as e: 
#         print(e)
        return(TLd,dTLd)

    return (TLd,dTLd,d2TLd)


def q_desired_direction(state):

    I_TLd = state.I_TLd
    
    qd = - I_TLd * (mt(I_TLd) @ I_TLd) **-0.5
    
    try:
        dI_TLd = state.dI_TLd
        dqd = (- dI_TLd * (mt(I_TLd) @ I_TLd) **-0.5
                + I_TLd * (mt(I_TLd) @ dI_TLd) * (mt(I_TLd) @ I_TLd)**-1.5)
        oqd = skew(qd) @ dqd
    except Exception as e: 
#         print(e)
        return (qd)

    try:
        d2I_TLd = state.d2I_TLd
        d2qd = (
            - d2I_TLd * (mt(I_TLd) @ I_TLd)**-0.5
            - dI_TLd * -0.5 * 2 * (mt(I_TLd) @ dI_TLd) * (mt(I_TLd) @ I_TLd)**-1.5
            + dI_TLd * (mt(I_TLd) @ dI_TLd) * (mt(I_TLd) @ I_TLd)**-1.5
            + I_TLd * (mt(dI_TLd) @ dI_TLd + mt(I_TLd) @ d2I_TLd) * (mt(I_TLd) @ I_TLd)**-1.5
            + I_TLd * (mt(I_TLd) @ dI_TLd) * -1.5 * 2 * (mt(I_TLd) @ dI_TLd) * (mt(I_TLd) @ I_TLd)**-2.5
        )
        doqd = skew(qd) @ d2qd
    except Exception as e: 
#         print(e)
        return (qd, dqd, oqd)

    
    return (qd, dqd, oqd, d2qd, doqd)



def q_feedback(q,oq,qd,oqd,doqd):
    
    dq = skew(oq) @ q
    
    e_q = cross(qd,q,axis=-2)
    e_oq = oq + skew(q) @ skew(q) @ oqd
    
    inner_q_oqd = sum(q*oqd, axis=-2, keepdims=True)
    
    q_actuation = ( - kq*e_q - koq*e_oq - inner_q_oqd*dq - skew(q) @ skew(q) @ doqd)    
    
    return (q_actuation, e_q, e_oq)



def qR_desired(t,Fd):

    # Fd = - T r3
    
    r3 = -Fd
    r2 = cross(r3,e1,axis=-2)
    r1 = cross(r2,r3,axis=-2)
    
    qRd = concatenate([r1,r2,r3],axis=-1)
    qRd = qRd / norm(qRd, axis=-2, keepdims=True)
    
    
    qod = 0 * Fd
    qtaud = 0 * Fd

    return (qRd,qod,qtaud)


# def ui_perp_feedback(q,qd,oq,oqd,Doqd,a):
#     # T. Lee paper - PD controller in S(2)

#     e_q = cross(qd,q,axis=-2)
#     e_oq = oq + skew(q) @ skew(q) @ oqd
    
#     dq = skew(oq) @ q
    
#     dot_q_oqd = sum(q*oqd, axis=1, keepdims=True)

#     q_feedback = ( - kq*e_q - koq*e_oq - dot_q_oqd*dq - skew(q)@skew(q) @ Doqd)
           
#     ui_perp = mq * lq * skew(q) @ q_feedback - mq * skew(q)@skew(q) @ a 
    
#     return ui_perp



def tau_feedback(R,o,Rd,od,taud):
#     # Mellinger paper - PD controller in SO(3)
    Rd = Rd.reshape(shape(R))
    od = od.reshape(shape(o))
    taud = taud.reshape(shape(o))
    
    # Mellinger paper - PD controller in SO(3)
    e_R = 1/2 * unskew(mt(Rd) @ R - mt(R) @ Rd)
    e_o = (o - od)
     
    M = J @ ( - kqR*e_R - kqo*e_o + taud) + skew(o)@J@o

    return M



# def T_feedback(x):
# #     # Mellinger paper - PD controller in SO(3)
# #     e_R = 1/2 * unskew(Rd.T @ R - R.T @ Rd).reshape(-1,1)
# #     e_o = (o - od)
    
# #     M = I @ ( - kr*e_R - ko*e_o + taud)

#     return T


def process_state(t,y):
    
    state = SimpleNamespace()
    
    state.t = t
    
    (p,v,R,o,delta_TLd,q,oq,qR,qo,V) = unpack_solution(y)
    
    state.p = p
    state.v = v
    state.delta_TLd = delta_TLd
    state.q = q
    state.oq = oq
    state.qR = qR
    state.qo = qo
    state.V = V
    
    state.pd = pr(t).reshape(shape(p))
    state.dpd = dpr(t).reshape(shape(p))
    state.d2pd = d2pr(t).reshape(shape(p))
    state.d3pd = d3pr(t).reshape(shape(p))
    state.d4pd = d4pr(t).reshape(shape(p))
#     state.d5pd = d5pr(t).reshape(shape(p))
        
    # Load R(3) controller
  
    Fd = F_feedback(state)    
    
    state.Fd = Fd
    
    # Retrieve TLd - Cable tensions
    
    (P, G) = matrix_nullspace()
    
    # Minimum norm TLd
    TLd_min = (P.T @ inv(P @ P.T) @ Fd).reshape(shape(delta_TLd))
    state.TLd_min = TLd_min
    
    # Repulsion for TL/cables not to be too close to each other 
    
    ddelta_TLd = TL_repulsive(TLd_min + delta_TLd) - TL_damping * delta_TLd
    
    if(isnan(ddelta_TLd).any()):
        print('NaN @ t = ', t)
    
    # Project dTLd_star on null space of P 
    ddelta_TLd = (G @ G.T @ ddelta_TLd.reshape(-1,n_cables*n_dims,1)).reshape(shape(delta_TLd))
        
    state.ddelta_TLd = ddelta_TLd
    
    
    # Compute desired cable directions
    
    # TODO: Include TLd derivatives    
    (TLd) = cable_tensions(state)

    # TODO / DEBUG: Change for other method
    (TLd_min, TLd_opt) = compute_TL(atleast_1d(t),Fd)
    TLd = TLd_min
    
    state.TLd = TLd
    state.I_TLd = TLd
    
    (qd) = q_desired_direction(state)
    state.qd = qd
    
    # Auxiliar variable - mu is projection of mud on current cable directions
    mud = qd @ mt(qd) @ TLd
    state.mud = mud
    
    mu = q @ mt(q) @ TLd
    state.mu = mu
    
    # Load dynamics
    
    dp = v
    dR = 0*R

    # Ideal actuation (actuation on "cable" level)
    
    dv = Fd / mL + g*e3
    do = 0*o    
    
    # Real actuation - Actuation is total force at quadrotor position
    
    F = sum(mu, axis=-3, keepdims=True)
    
    state.F = F
    
    dv = F / mL + g*e3
    do = 0*o
    
    state.dv = dv
    
    (Fd,dFd) = F_feedback(state)
    state.dFd = dFd

    dTLd_min = (P.T @ inv(P @ P.T) @ dFd).reshape(shape(delta_TLd))
    state.dTLd_min = dTLd_min 
    
    (_, dTLd) = cable_tensions(state)
    state.dTLd = dTLd
    state.dI_TLd = dTLd
    
    (_,dqd,oqd) = q_desired_direction(state)
    state.dqd = dqd
    state.oqd = oqd
    
    dq = skew(oq) @ q
    state.dq = dq
    
    # Auxiliar variable - mu is projection of mud on current cable directions
    dmud = (dqd @ mt(qd) @ TLd
            + qd @ mt(dqd) @ TLd
            + qd @ mt(qd) @ dTLd)
    state.dmud = dmud
    
    dmu = (dq @ mt(q) @ TLd
          + q @ mt(dq) @ TLd
          + q @ mt(q) @ dTLd)
    state.dmu = dmu
       
    dF = sum(dmu, axis=-3, keepdims=True)
    
    state.dF = dF
    
    d2v = dF / mL
    
    state.d2v = d2v
    
    (Fd,dFd,d2Fd) = F_feedback(state)
    state.d2Fd = d2Fd
        
    # TODO
    d2delta_TLd = 0*ddelta_TLd
    state.d2delta_TLd = d2delta_TLd
    
    d2TLd_min = (P.T @ inv(P @ P.T) @ d2Fd).reshape(shape(delta_TLd))
    state.d2TLd_min = d2TLd_min 
    (_,_,d2TLd) = cable_tensions(state)
    state.d2TLd = d2TLd
    state.d2I_TLd = d2TLd
    (_,_,_,d2qd,doqd) = q_desired_direction(state)
    state.d2qd = d2qd
    state.doqd = doqd
    
    state.dp = dp
    state.dv = dv
    
    state.dR = dR
    state.do = do
    
    # Cable dynamics
    
    (q_actuation, e_q, e_oq) = q_feedback(q,oq,qd,oqd,doqd)
    
    state.q_actuation = q_actuation
    state.e_q = e_q
    state.e_oq = e_oq
    
    a = dv - g*e3
    
    u_parallel = mu + mQ * lq * norm(oq, axis=-2, keepdims=True)**2 * q + mQ * q @ mt(q) @ a
    
    # Actuation without cross-terms
    u_perp = mQ * lq * skew(q) @ q_actuation - mQ * skew(q)@skew(q) @ a 

    pd = pr(t)
    dpd = dpr(t)
    d2pd = d2pr(t)
    d3pd = d3pr(t)

    pe = p-pd
    ve = v-dpd
    ae = dv-d2pd

    zoq = (
        - oq 
        - skew(q) @ skew(q) @ oqd
        - 1 / L_q * (mt(qd) @ TLd) * skew(q) @ (
            1/mL * (x_pv * pe + ve))
    )
    state.zoq = zoq

    u_perp_d =  (mQ * lq) * skew(q) @ (
        - 1 / lq * skew(q) @ a 
        - skew(dq) @ skew(q) @ oqd
        - skew(q) @ skew(dq) @ oqd
        - skew(q) @ skew(q) @ doqd
        - 1 / L_q * (
                + (mt(dqd) @ TLd) * skew(q)
                + (mt(qd) @ dTLd) * skew(q)
                + (mt(qd) @ TLd) * skew(dq)
            ) @ (
                    1/mL * (x_pv * pe + ve)
                )
        - 1 / L_q * (mt(qd) @ TLd) * skew(q) @ (
            1/mL * (x_pv * ve + ae)
        ) 
        + koq / L_oq * zoq
        + L_q / L_oq * skew(q) @ qd
    )
    
    # Actuation with cross-terms
    if not NO_U_PERP_CROSS_TERMS:
        u_perp = u_perp_d
    
    
    state.u_perp_d = u_perp_d
    
    state.a = a
    state.u_parallel = u_parallel
    state.u_perp = u_perp

    u = u_perp + u_parallel
    state.u = u
    
    doq = 1 / lq * skew(q) @ a - 1 / (mQ * lq) * skew(q) @ u_perp

    state.doq = doq
    
    # Quadrotor dynamics
    
    # DEBUG: Independent Rd control
    (qRd,qod,qtaud) = qR_desired(t,u)
    
    qtau = tau_feedback(qR,qo,qRd,qod,qtaud)
    
    dqR = qR @ skew(qo)
    dqo = inv_J @ qtau -  inv_J @ skew(qo)@J@qo
    
    state.qRd = qRd
    state.qod = qod
    state.qtaud = qtaud
    state.qtau = qtau
    state.dqR = dqR
    state.dqo = dqo
    
    qFd = u
    r3 = qR @ e3
    qT = - mt(r3) @ qFd
    
    state.qFd = qFd
    state.qT = qT
    state.qF = qT * - r3
    


    # Lyapunov verification
    
    de_q = cross(dqd,q,axis=-2) + cross(qd,dq,axis=-2)
    state.de_q = de_q
    
    dV = (
        - kp*x_pv *  mt(pe) @ (pe) 
        - kv*x_pv * mt(pe) @ (ve)
        - (kv - x_pv) * mt(ve) @ ve 
        - koq * sum( mt( zoq ) @ ( zoq )
        , axis=-3, keepdims=True)
    )
    
    state.dV = dV

    # Output time derivative as 1-d array and not a 2D nx1 array
    
#     print(state.dp.shape)
#     print(state.dv.shape)
#     print(state.dR.shape)
#     print(state.do.shape)
#     print(state.ddelta_TLd.shape)
#     print(state.dq.shape)
#     print(state.doq.shape)
#     print(state.dqR.shape)
#     print(state.dqo.shape)
#     print(state.dV.shape)
    
    dy = pack_solution(state.dp,
                       state.dv,
                       state.dR,
                       state.do, 
                       state.ddelta_TLd, 
                       state.dq, 
                       state.doq, 
                       state.dqR,
                       state.dqo,
                       state.dV,
                      ).ravel()
    
    # if atleast_1d(t % 1 < 1e-3).all():
        # print(f"t = {t}")

    return dy, state
    


# ### Use optimization to solve for TL

# In[ ]:


# Find TL solution with 1) minimum norm and 2) that observes the angle restrictions 

TL_len = n_cables*n_dims

def compute_TL(t,Fd):
    
    (P,G) = matrix_nullspace()
    
    TL_min = (P.T @ inv(P @ P.T) @ Fd).reshape(-1,n_cables,n_dims,1)
    TL_opt = empty_like(TL_min)
    
    for i, ti in enumerate(t):
        Fdi = Fd[i,...]
        
        TL_mini = TL_min[i].reshape(n_cables*n_dims,1)
               
        ### Setup optimization
        
        Q = eye(TL_len)

        # minimum angle with inertial/body z-axis
        theta = theta_min_for_optimization

        A = P
        b = Fdi.ravel()

        # TL numbering starts with 1 and is clockwize from x-axis
        #
        # TL_1 (NE)
        # TL_2 (NW)
        # TL_3 (SW)
        # TL_4 (SE)

        # Inner products with these vectors with TL must be positive for 
        # some combinations of vector/TL
        #
        # Vectors are in the body frame as when the box rotates 
        # the cones should also rotate
        
        v_N = [ cos(theta),            0,   sin(theta)] # E
        v_E = [          0,   cos(theta),   sin(theta)] # N
        v_S = [-cos(theta),            0,   sin(theta)] # W
        v_W = [          0,  -cos(theta),   sin(theta)] # S
        
        # Match "rho" in SCORE lab
        Ai = vstack([
                concatenate([v_N , zeros(9)]),
                concatenate([v_W, zeros(9)]),
                concatenate([zeros(3) , v_N, zeros(6)]),
                concatenate([zeros(3) , v_E , zeros(6)]),
                concatenate([zeros(6) , v_S, zeros(3)]),
                concatenate([zeros(6) , v_W, zeros(3)]),
                concatenate([zeros(9) , v_S]),
                concatenate([zeros(9) , v_E]),
                ])

        ##   Comment to choose -> theta angle with inertial z-axis
        ## Uncomment to choose -> theta angle with body z-axis
        # Ai = (R_mult.T @ Ai.T).T

        bi = zeros(Ai.shape[0])
        
        ### Run optimization

        # use solution from previous timestep
        try:
            x0 = compute_TL.x0
        except AttributeError:
            x0 = random.randn(TL_len)

        def loss(x, sign=1.):
            return sign * (0.5 * x.T @ Q @ x)

        def jac(x, sign=1.):
            return sign * (x.T @ Q)

        cons = ({'type':'eq',
                'fun':lambda x: A @ x - b,
                 'jac':lambda x: A},
#                {'type':'ineq',
#                 'fun':lambda x: Ai @ x - bi,
#                  'jac':lambda x: Ai}
               )

        options = {'disp':False}

        res = optimize.minimize(loss, 
                                x0, 
                                jac=jac,
                                constraints=cons,
                                options=options,
                                tol=1e-9)

        if(res.success):
#             print(res)
            TL_opt[i] = res.x.reshape(n_cables,n_dims,1)
            compute_TL.x0 = res.x
        else:
            print(f't = {ti:0.3} \t No solution:', res.message)
#             TL_opt[i] = zeros_like(res.x)

    return (TL_min, TL_opt)


# ### Initial states

# In[ ]:


# Quadrotor initial states

p0 = zeros((3,1))

v0 = zeros((3,1))

# R0 = expm(skew(zeros((3,1))))
R0 = Rx(10*pi/180) # @ Ry(5*pi/180)

o0 = zeros((3,1))

delta_TLd0 = zeros((n_cables*n_dims,1))

state0 = SimpleNamespace()

state0.p = p0
state0.v = v0
state0.pd = pr(t0)
state0.dpd = dpr(t0)
state0.d2pd = d2pr(t0)

state0.R = eye(3)
state0.o = zeros((3,1))
state0.Rd = eye(3)
state0.od = zeros((3,1))
state0.taud = zeros((3,1))

# # Initialize delta_TL0 at feasible location (outside of potential singularity)
Fd0 = F_feedback(state0)
B_Fd0 = R0.T @ Fd0
Md0 = zeros((3,1))
B_FdMd0 = vstack([B_Fd0,Md0])

(TLd_min0, TLd_opt0) = compute_TL(
    [t0],
    Fd0.reshape(1,n_dims,1))

# delta_TLd0 = (B_TLd_opt0 - B_TLd_min0).reshape(-1,n_cables,n_dims,1)


# # TODO: choose q0
# d = ones([3,1]) / sqrt(3)
# q0 = tile(d,n_cables).T.reshape(n_cables,n_dims,1)

q0 = tile(e3,n_cables).T.reshape(n_cables,n_dims,1)

# Strings point outwards of center of mass
q0 = array([ [[ 0], [0], [1]] ])

oq0 = zeros((n_cables, n_dims,1))

# qR0 = tile(eye(3),n_cables).T.reshape(n_cables,3,3)
qR0 = tile(Rx(0*pi/180),n_cables).T.reshape(n_cables,3,3)

qo0 = zeros((n_cables, n_dims,1))

V0 = array(0)


# In[ ]:


# pd = pr(t0)
# dpd = dpr(t0)
# d2pd = d2pr(t0)

# pd = pd.reshape(shape(p0))
# dpd = dpd.reshape(shape(p0))
# d2pd = d2pd.reshape(shape(p0))

# # PD controller
# F = m * (- kp*(p0-pd) 
#          - kv*(v0-dpd) 
#          - g*e3 
#          + d2pd)


# ### Auxiliar functions for packing and unpacking the system state

# In[ ]:


def pack_state(p,v,R,o,delta_TLd,q,oq,qR,qo,V):
#     print(p.shape)
#     print(v.shape)
#     print(R.reshape(-1,1).shape)
#     print(o.shape)
#     print(delta_TLd.reshape(-1,1).shape)
    return vstack([p,
                   v,
                   R.reshape(-1,1),
                   o,
                   delta_TLd.reshape(-1,1),
                   q.reshape(-1,1),
                   oq.reshape(-1,1),
                   qR.reshape(-1,1),
                   qo.reshape(-1,1),
                   V])

state_sizes = [n_dims,      # p
               n_dims,      # v
               3*3,    # R
               n_dims,      # o
               n_dims*n_cables,   # TLd
               n_dims*n_cables,   # q
               n_dims*n_cables,   # oq
               3*3*n_cables,    # quadrotor R
               n_dims*n_cables,   # quadrotor o
               1,   # V
              ]
state_idx = cumsum(state_sizes)

def unpack_state(x):
    (p,v,R,o,delta_TLd,q,oq,qR,qo,V,_) = split(x,state_idx)
    
    return (p.reshape(n_dims,1),
            v.reshape(n_dims,1),
            R.reshape(3,3),
            o.reshape(n_dims,1),
            delta_TLd.reshape(n_cables,n_dims,1),
            q.reshape(n_cables,n_dims,1),
            oq.reshape(n_cables,n_dims,1),
            qR.reshape(n_cables,3,3),
            qo.reshape(n_cables,n_dims,1),
            V)


# In[ ]:


# # TEST - Unpack(Pack(x)) = x
# orig = (p0,v0,R0,o0,delta_TLd0,q0,oq0,qR0,qo0,V0)
# state0 = pack_state(p0,v0,R0,o0,delta_TLd0,q0,oq0,qR0,qo0,V0)
# (p0,v0,R0,o0,delta_TLd0,q0,oq0,qR0,qo0,V0) = unpack_state(state0)
# (sum(abs(orig[0] - p0)) + 
# sum(abs(orig[1] - v0)) + 
# sum(abs(orig[2] - R0)) + 
# sum(abs(orig[3] - o0)) +
# sum(abs(orig[4] - delta_TLd0)) +
# sum(abs(orig[5] - q0)) +
# sum(abs(orig[6] - oq0)) +
# sum(abs(orig[7] - qR0)) +
# sum(abs(orig[8] - qo0)) +
# sum(abs(orig[9] - V0)))


# ## Integration setup (build f, y, and args)

# In[ ]:


from scipy.integrate import ode

y0 = pack_state(p0,
                v0,
                R0.ravel(),
                o0,
                delta_TLd0,
                q0,
                oq0,
                qR0,
                qo0,
                V0,
               ).ravel()
t0 = 0

# y' = f(t,y)
def f(t, y):
    
    dy, _ = process_state(t,y)
    
    return dy


# In[ ]:


# from functools import partial

# # Define fixed arguments for 'f' used for ODE 
# f_ode = partial(f,args=args)
f_ode = f


# ### Auxiliar function to unpack a complete solution by states

# In[ ]:


def unpack_solution(x):
    
#     import pdb; pdb.set_trace()
    
    try:
        (p,v,R,o,delta_TLd,q,oq,qR,qo,V,_) = split(x,state_idx,axis=-1)
    except IndexError:
        (p,v,R,o,delta_TLd,q,oq,qR,qo,V,_) = split(x,state_idx)
        
    return (p.reshape(-1,1,n_dims,1),
            v.reshape(-1,1,n_dims,1),
            R.reshape(-1,1,3,3),
            o.reshape(-1,1,n_dims,1),
            delta_TLd.reshape(-1,n_cables,n_dims,1),
            q.reshape(-1,n_cables,n_dims,1),
            oq.reshape(-1,n_cables,n_dims,1),
            qR.reshape(-1,n_cables,3,3),
            qo.reshape(-1,n_cables,n_dims,1),
           V.reshape(-1,1,1,1))

def pack_solution(p,v,R,o,delta_TLd,q,oq,qR,qo,V):
    r = array(R.shape)
    r[-2] = 9
    r[-1] = 1
    
    s = array(q.shape)
    s[-3] = 1
    s[-2] = n_cables*3
    s[-1] = 1
    
    sR = array(q.shape)
    sR[-3] = 1
    sR[-2] = n_cables*9
    sR[-1] = 1
    
    t = [1]*p.ndim
    t[0] = s[0]
    
    x = R.reshape(r)
    
#     print(shape(p))
#     print(shape(v))
#     print(shape(R.reshape(r)))
#     print(shape(o))
#     print(shape(delta_TLd.reshape(s)))
#     print(shape(q.reshape(s)))
#     print(shape(oq.reshape(s)))
#     print(shape(atleast_2d(V).reshape(t)))
#     print('----------')
    
    return concatenate([p,
                        v,
                        R.reshape(r),
                        o,
                        delta_TLd.reshape(s),
                        q.reshape(s),
                        oq.reshape(s),
                        qR.reshape(sR),
                        qo.reshape(s),
                        atleast_2d(V).reshape(t)],axis=-2)
