#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.environ["GEOMETRY_CONFIGURATION"] = "/home/score/ws_multi_quadrotor/src/pyquad/src/geometry_rigid_body_4_simulation.yaml"
# os.environ["GEOMETRY_CONFIGURATION"] = "/home/score/ws_multi_quadrotor/src/pyquad/src/geometry_rigid_body_4_experimental.yaml"
configuration_geometry = os.environ["GEOMETRY_CONFIGURATION"]


# ## Imports

# In[2]:


# import numpy as np
from numpy import *

from scipy import optimize
from scipy.integrate import solve_ivp, cumtrapz
from scipy.linalg import null_space, expm, inv, norm, eig, pinv
from scipy.spatial.transform import Rotation

from types import SimpleNamespace

import inspect

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

from cvxopt import matrix, solvers


# ### Math and Physics constants

# In[3]:


g = 9.8

e1 = array([[1,0,0]]).T
e2 = array([[0,1,0]]).T
e3 = array([[0,0,1]]).T


# ### Geometry variables

# In[4]:


# Todo: Define rho then get n_cables and n_dims from rho matrix
n_cables = 4
n_dims = 3

# TODO: Different cable lengths for different cables
cable_length = 1.23

# Array of cable lengths
lq = cable_length * ones(n_cables).reshape(-1,1,1)

# Load - Rigid body dimensions
length = 1.2
width = 0.8
height = .2

# Matrix of attachment points in body frame
# Change also --> q0, Ai

# Quadrotor enumetation starts at N and goes clockwise (E,S,W)
rho = array([ [[ length/2], [ width/2], [-height/2]],
              [[-length/2], [ width/2], [-height/2]],
              [[-length/2], [-width/2], [-height/2]],
              [[ length/2], [-width/2], [-height/2]] ])

POINT_MASS = False


# ## Simulation parameters

# #### For Lyapunov verification

# In[5]:


# Integration times

t0 = 0
tf = 10

# DEBUG selectors

# For a complete Lyapunov function with negative semi-definite time derivative 
NO_U_PERP_CROSS_TERMS = False

# TLd derivatives are zero (emulate solution of optimization problem)
TLD_DERIVATIVES_ZERO = False

# Uses optimization solution to find TLd
USE_OPTIMIZATION = False

# Remove instability causing terms due to model uncertainty
REMOVE_INSTABILITY_TERMS = False

# Use quadrotor actuation dynamics
SIMULATE_REAL_QUADROTOR = False

# Don't cancel ou skew(o) I o in Md
IGNORE_OMEGA_CANCELATION = False

# Used to auxiliar variables from the integration results
# Plot only: 1e-2
# Lyapunov verify: 1e-4
t_step = 1e-4

# Tolerance for ODE solver
tolerance = 1e-8


# #### For more realistic model

# In[6]:


# Integration times

t0 = 0
tf = 10

# DEBUG selectors

# For a complete Lyapunov function with negative semi-definite time derivative 
NO_U_PERP_CROSS_TERMS = False

# TLd derivatives are zero (emulate solution of optimization problem)
TLD_DERIVATIVES_ZERO = True

# Uses optimization solution to find TLd
USE_OPTIMIZATION = True

# Remove instability causing terms due to model uncertainty
REMOVE_INSTABILITY_TERMS = False

# Use quadrotor actuation dynamics
SIMULATE_REAL_QUADROTOR = False

# Don't cancel ou skew(o) I o in Md
IGNORE_OMEGA_CANCELATION = False

# Used to auxiliar variables from the integration results
# Plot only: 1e-2
# Lyapunov verify: 1e-4
t_step = 1e-2

# Tolerance for ODE solver
tolerance = 1e-4


# ### Control gains

# In[7]:


Kp = 1/3 * diag([1.1,1.2,1.3])
Kv = 1/3 * diag([1.1,1.2,1.3])

X_pv = (Kp + Kv) / 4

L_q = 15
L_oq = 25

kq = 2
koq = 5

L_qR = 100
L_oqR = 10

kqR = 10
kqo = 10

L_a = 0
kr = 0 
ko = 0

###

# DEBUG
TL_repulsive_gain = 0.0
TL_damping = 0.0

mL = 1.200
mass_quad = 1.500
mQ = mass_quad * ones([n_cables,1,1])

I = 1/12 * mL * diag([width**2 + height**2, length**2 + height**2, length**2 + width**2])

# inertia matrix for the quadrotors
J = diag([0.21, 0.22, 0.23])
inv_J = inv(J)

J = stack(n_cables*[J])
inv_J = stack(n_cables*[inv_J])

# Minimum angle with vertical of z-body direction when optimizing
theta_min_for_optimization = deg2rad(-10)


# ### Load parameters from file

# #### Geometry

# In[8]:


try:
    import yaml

    with open(configuration_geometry) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfg = yaml.load(file)

        
    # Geometry 
    n_cables = cfg['number_of_vehicles']
    n_dims = cfg['dimensions']

    rho = np.array([cfg[i]['attachment_point'] for i in range(cfg['number_of_vehicles'])]).reshape(cfg['number_of_vehicles'],cfg['dimensions'],1)

    lq = np.array([cfg[i]['cable_length'] for i in range(cfg['number_of_vehicles'])]).reshape(cfg['number_of_vehicles'],1,1)

    mQ = np.array([cfg[i]['mass'] for i in range(cfg['number_of_vehicles'])]).reshape(cfg['number_of_vehicles'],1,1)

    # Load - Rigid body dimensions
    length = cfg['load']['length']
    width = cfg['load']['width']
    height = cfg['load']['height']

    mL = cfg['load']['mass']
    I = cfg['load']['inertia']
    
    # inertia matrix for the quadrotors
    # FIXME: read all quadrotors
    J = stack([np.array(cfg[i]['inertia']) for i in range(cfg['number_of_vehicles'])])
    inv_J = stack([inv(np.array(cfg[i]['inertia'])) for i in range(cfg['number_of_vehicles'])])
    
except e as Exception:
    print(f"Couldn't read all configurations from file: {configuration_geometry}")
    print(e)
    pass


# #### Simulation parameters

# In[9]:


try:
    import yaml

    with open(configuration_geometry) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfg = yaml.load(file)

    t0 = cfg['t0']
    tf = cfg['tf']

    t_step = cfg['t_step']
    tolerance = cfg['tolerance']
    
    POINT_MASS = cfg['POINT_MASS']
    
    NO_U_PERP_CROSS_TERMS = cfg['NO_U_PERP_CROSS_TERMS']
    TLD_DERIVATIVES_ZERO = cfg['TLD_DERIVATIVES_ZERO']
    USE_OPTIMIZATION = cfg['USE_OPTIMIZATION']

    REMOVE_INSTABILITY_TERMS = cfg['REMOVE_INSTABILITY_TERMS']

    SIMULATE_REAL_QUADROTOR = cfg['SIMULATE_REAL_QUADROTOR']

except Exception as e:
    print(f"Couldn't read all configurations from file: {configuration_geometry}")
    print(e)
    pass


# #### Controller gains

# In[10]:


try:
    import yaml

    with open(configuration_geometry) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfg = yaml.load(file)

    Kp = np.array(cfg['controller']['Kp'])
    Kv = np.array(cfg['controller']['Kv'])
    X_pv = np.array(cfg['controller']['X_pv'])

    L_q = cfg['controller']['L_q']
    L_oq = cfg['controller']['L_oq']

    kq = cfg['controller']['kq']
    koq = cfg['controller']['koq']

    L_qR = cfg['controller']['L_qR']
    L_oqR = cfg['controller']['L_oqR']

    kqR = cfg['controller']['kqR']
    kqo = cfg['controller']['kqo']

    L_a = cfg['controller']['L_a']
    kr = cfg['controller']['kr'] 
    ko = cfg['controller']['ko']
    
    TL_damping = cfg['controller']['TL_damping']
    TL_repulsive = cfg['controller']['TL_repulsive']

    theta_min_for_optimization = np.deg2rad(cfg['controller']['theta_min_for_optimization'])

except Exception as e:
    print(f"Couldn't read all configurations from file: {configuration_geometry}")
    print(e)
    pass


# ### Save gains and parameters for future analysis

# In[11]:


# Save to yaml file with date


# ## Load trajectory

# In[12]:


import sympy as sp
from sympy.utilities.autowrap import autowrap
from sympy.utilities.autowrap import ufuncify

t = sp.symbols('t', real=True)

r,rx,ry, rz = sp.symbols('r r_x r_y r_z')
omega,omega_x,omega_y = sp.symbols('omega omega_x omega_y')
theta_x,theta_y = sp.symbols('theta_x theta_y')
omega,omega_x,omega_y = sp.symbols('omega omega_x omega_y')
theta, theta_x,theta_y = sp.symbols('theta theta_x theta_y')
cx, cy, cz = sp.symbols('cx cy cz')
alpha = sp.symbols('alpha')

### Trajectory parameters

# Lemniscate
args = {rx:0.75,
        ry:1.5,
        rz: 0.2,
        omega: sp.S(1)/2,
        theta: sp.pi/4, 
        cx: 0.0,
        cy: 0.0,
        cz: -1.0,
       }


# Trajectory definition

# Circle / Oval / Lemniscate
p_symb = sp.Matrix([cx + rx*sp.sin(2*omega*(t+theta)),
                    cy + ry*sp.sin(omega*(t+theta)),
                    cz + rz*sp.cos(omega*(t+theta)) ]) 

# # Straight line - almost constant
# p_symb = sp.Matrix([ 0.001*t,
#                      0.0,
#                     -1.0])


# In[13]:


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


# ### Load attitude reference

# In[14]:


def unskew(M):
    # TODO: Check if M is anti-symmetric
    # TODO: Check that unskew(skew(v)) == v
    return sp.Matrix([-M[1,2], M[0,2], -M[0,1], ])

# # TODO: Improve generation for trajectories with z(t) not constant
# dp = sp.simplify(p_symb.diff(t).subs(args))
# dir = dp / dp.norm()

dir = p_symb.diff(t).subs(args)
dir[2] = 0
dir = sp.simplify(dir)
dir = dir / dir.norm()

# FIXME: DEBUG only
# Pointing north
dir = sp.Matrix(e1)

# x-axis of body frame points in velocity direction of the trajectory (tangent to trajectory)
r1 = dir
r2 = - dir.cross(sp.Matrix(e3))
r3 = dir.cross(r2)

r1 = r1 / r1.norm()
r2 = r2 / r2.norm()
r3 = r3 / r3.norm()

r1 = sp.simplify(r1)
r2 = sp.simplify(r2)
r3 = sp.simplify(r3)
    
R_symb = sp.Matrix([r1.T,r2.T,r3.T]).T

omega_symb = sp.simplify(unskew(R_symb.T @ R_symb.diff(t)))

tau_symb = omega_symb.diff(t)
dtau_symb = omega_symb.diff(t,2)
d2tau_symb = omega_symb.diff(t,3)


# In[15]:


Rr_vec = stack([ufuncify(t,R_symb[i].subs(args)) for i in arange(len(R_symb))])
omegar_vec = stack([ufuncify(t,omega_symb[i].subs(args)) for i in arange(n_dims)])
taur_vec = stack([ufuncify(t,tau_symb[i].subs(args)) for i in arange(n_dims)])
dtaur_vec = stack([ufuncify(t,dtau_symb[i].subs(args)) for i in arange(n_dims)])
d2taur_vec = stack([ufuncify(t,d2tau_symb[i].subs(args)) for i in arange(n_dims)])

# TODO: Fix

# def Rr(t):
#     return stack(len(atleast_1d(t))*[eye(3)])

# def omegar(t):
#     return stack(len(atleast_1d(t))*[zeros([3,1])])

# def taur(t):
#     return stack(len(atleast_1d(t))*[zeros([3,1])])

def Rr(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([Rr_vec[i](x) for i in range(len(Rr_vec))],axis=1).reshape(-1,1,3,3)

    # If first dimension is singleton then remove it
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec

def omegar(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([omegar_vec[i](x) for i in range(len(omegar_vec))],axis=1).reshape(-1,1,3,1)

    # If first dimension is singleton then remove it
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])

    return vec

def taur(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([taur_vec[i](x) for i in range(len(taur_vec))],axis=1).reshape(-1,1,3,1)
    
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec

def dtaur(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([dtaur_vec[i](x) for i in range(len(dtaur_vec))],axis=1).reshape(-1,1,3,1)
    
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec

def d2taur(x_orig):
    x = asarray(x_orig).reshape(-1,1)
    
    vec = stack([d2taur_vec[i](x) for i in range(len(d2taur_vec))],axis=1).reshape(-1,1,3,1)
    
    if(vec.shape[0]==1):
        vec = vec.reshape(vec.shape[2:])
    
    return vec


# ### Auxiliar functions

# In[16]:


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


# In[17]:


def mt(M):
# Matrix transpose no matter what is M shape.  M = (...) x n x m or M = (...) x m x n
    
    return np.swapaxes(M,-1,-2)


# In[18]:


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


# In[19]:


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

#     assert(isRotationMatrix(R))

    sy = np.sqrt(R[...,0,0] * R[...,0,0] +  R[...,1,0] * R[...,1,0])

    singular = sy < 1e-6

    x = array(np.arctan2(R[...,2,1] , R[...,2,2]))
    y = array(np.arctan2(-R[...,2,0], sy))
    z = array(np.arctan2(R[...,1,0], R[...,0,0]))
    
    
    x[singular] = np.arctan2(-R[singular,1,2], R[singular,1,1])
    y[singular] = np.arctan2(-R[singular,2,0], sy[singular])
    z[singular] = 0

    return np.array([x, y, z])


# In[20]:


def check_derivative(f,df,t,tol=tolerance):
#     int_df = cumtrapz(df,t,axis=0,initial=0)
#     max_error = amax(abs(f-int_df - (f[0]-int_df[0])))
#     if(not SIMULATE_REAL_QUADROTOR):
#         if(max_error > tol):
#             caller = inspect.getframeinfo(inspect.stack()[1][0])
#             print(f"High derivative error: tol < {max_error:.2E} \t Line {caller.lineno:3}: {caller.code_context}")
    return

def check_equality(arg1,arg2,tol=tolerance):
#     max_error = amax(abs(arg1-arg2))
#     if(not SIMULATE_REAL_QUADROTOR):
#         if(max_error > tol):
#             caller = inspect.getframeinfo(inspect.stack()[1][0])
#             print(f"High state error: tol < {max_error:.2E} \t Line {caller.lineno:3}: {caller.code_context}")
    return


# In[21]:


def F_vector_to_Rotation(F):
    F = F.reshape(-1,n_cables,3,1)

    # Angle of 
    angle = (np.arccos(mt(-e3) @ F / norm(F, axis=-2, keepdims = True)))
    vector = np.cross(-e3,F,axis=-2)
    vector = vector / norm(vector, axis=-2, keepdims = True)

#     print(Rotation.from_rotvec((angle * vector).reshape(-1,3)))
    
    return Rotation.from_rotvec((angle * vector).reshape(-1,3))
    


# ## Problem definition

# ### Contruction of P matrix and its nullspace

# In[22]:


def matrix_nullspace(R=eye(3)):
    # Computes P matrix and its nullspace basis G for a certain load attitude
    
    # TLd = [TLd_11, TLd_12, .... , TLd_nn-1 , TLd_nn]

    # Sum of cable tensions in each axis
    P = kron( ones(n_cables), eye(n_dims))

    P = vstack([
        P,
        hstack([skew(R @ rho[i]) for i in range(n_cables)])   # Only care about load yaw control (not pitch or roll)
    ])

    # G is subspace of allowable TLd derivatives (which do not affect F or M)
    G = null_space(P)

    return (P, G)


# ### Load PD control functions

# In[23]:


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


# In[24]:


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
    F = mL * (- Kp @ (p-pd) 
             - Kv @ (v-dpd) 
             - g*e3 
             + d2pd)
    
    try:
        dv = state.dv
        d3pd = state.d3pd
        
        dF = mL * (- Kp @ (v-dpd) 
                 - Kv @ (dv-d2pd) 
                 + d3pd)
    except Exception as e: 
#         print(e)
        return F
    
    try:
        d2v = state.d2v
        d4pd = state.d4pd
        
        d2F = mL * (- Kp @ (dv-d2pd) 
                 - Kv @ (d2v-d3pd) 
                 + d4pd)
    except Exception as e: 
#         print(e)
        return (F,dF)
    
    return (F,dF,d2F)



def M_feedback(state):
#     import inspect
#     frame = inspect.currentframe()
#     for key,val in state.__dict__.items(): 
#         frame.f_back.f_locals[key]=val
    
    R = state.R
    o = state.o
    Rd = state.Rd
    od = state.od
    taud = state.taud
    
    # Mellinger paper - PD controller in SO(3)
    Re = mt(R) @ Rd
    e_R = 1/2 * unskew(mt(Re) - Re)
    e_o = (o - od)
    
    M = I @ ( - kr * e_R - ko * e_o + taud) + skew(o)@I@o
    
    if(IGNORE_OMEGA_CANCELATION):
        M = I @ ( - kr * e_R - ko * e_o + taud)

#     # Koditschek Lyapunov function
#     Re = mt(R) @ Rd
#     e_R = 1/2 * unskew(Re @ Q @ mt(Q) - Q @ mt(Q) @ mt(Re))
#     e_o = o - Re @ od
    
    try:
        do = state.do
        tau = do
        dtaud = state.dtaud
    
        dRe = - skew(o - Re @ od) @ Re
        de_R = 1/2 * unskew( mt(Re) @ skew(o - Re @ od) - mt(skew(o - Re @ od)) @ Re )
        de_o = tau - taud
        
        dM = I @ ( - kr * de_R - ko * de_o + dtaud) + skew(tau)@I@o + skew(o)@I@tau
    except Exception as e: 
#         print(e)
        return M
    
    try:
        dtau = state.d2o
        dod = taud
        d2taud = state.d2taud
        
        
        d2e_R = 1/2 * unskew( 
            mt(dRe) @ skew(o - Re @ od) + mt(Re) @ skew(do - dRe @ od - Re @ dod)  
             - mt(skew(do - dRe @ od - Re @ dod)) @ Re - mt(skew(o - Re @ od)) @ dRe 
        )
        d2e_o = dtau - dtaud

        d2M = I @ ( - kr * d2e_R - ko * d2e_o + d2taud) + skew(dtau)@I@o + skew(tau)@I@tau + skew(tau)@I@tau + skew(o)@I@dtau
    except Exception as e: 
#         print(e)
        return (M,dM)
    
    return (M,dM,d2M)



# Cable repulsive function
def TL_repulsive(TL):
    
    cable_matrix = (- TL / norm(TL, axis=-2, keepdims=True)) * lq
    
    quad_positions = - cable_matrix + rho

    dTL = - TL_repulsive_gain * Dlog_barrier(quad_positions)
    dTL[:,2] = 0    # No repulsion along z-axis

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
        if (TLD_DERIVATIVES_ZERO):
            dqd = 0*dqd
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
        if (TLD_DERIVATIVES_ZERO):
            d2qd = 0*d2qd
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



# FIXME: Use Fd --> Rd from other code
def qR_desired(t,Fd):

    # Fd = - T r3
    
    r3 = -Fd
    r2 = cross(r3,e1,axis=-2)
    r1 = cross(r2,r3,axis=-2)
    
    qRd = concatenate([r1,r2,r3],axis=-1)
    qRd = qRd / norm(qRd, axis=-2, keepdims=True)
    
    # FIXME:    
    qod = 0 * Fd
    qtaud = 0 * Fd

    return (qRd,qod,qtaud)



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


# ### Closed-loop system dynamics

# In[25]:


def process_state(t,y):
    
    state = SimpleNamespace()
    
    state.t = t
    
    (p,v,R,o,delta_TLd,q,oq,qR,qo,V) = unpack_solution(y)
    
    state.p = p
    state.v = v
    state.R = R
    state.o = o
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
    
    state.Rd = Rr(t).reshape(shape(R))
    state.od = omegar(t).reshape(shape(o))
    state.taud = taur(t).reshape(shape(o))
    state.dtaud = dtaur(t).reshape(shape(o))
    state.d2taud = d2taur(t).reshape(shape(o))
#     state.d3taud = d3taur(t).reshape(shape(o))
    
    # Load SE(3) controller
  
    Fd = F_feedback(state)
    Md = M_feedback(state)
    
#     print("Fd: ", Fd.shape)
#     print("Md: ", Md.shape)
    
    state.Fd = Fd
    state.Md = Md
    
    # Retrieve TLd - Cable tensions
    
    B_Fd = mt(R) @ Fd    
    B_FdMd = concatenate([B_Fd,Md],axis=-2)
    (P, G) = matrix_nullspace()
    
    state.B_Fd = B_Fd
    state.B_FdMd = B_FdMd
    
    # Minimum norm TLd
    TLd_min = (P.T @ inv(P @ P.T) @ B_FdMd).reshape(shape(delta_TLd))
    state.TLd_min = TLd_min
    
    # Repulsion for TL/cables not to be too close to each other 
    
    ddelta_TLd = TL_repulsive(TLd_min + delta_TLd) - TL_damping * delta_TLd
    
    if(isnan(ddelta_TLd).any()):
        print('NaN @ t = ', t)
    
    # Project dTLd_star on null space of P 
    ddelta_TLd = (G @ G.T @ ddelta_TLd.reshape(-1,12,1)).reshape(shape(delta_TLd))
        
    state.ddelta_TLd = ddelta_TLd
    
    
    # Compute desired cable directions
    
    # TODO: Include TLd derivatives    
    (TLd) = cable_tensions(state)
    
    # TODO / DEBUG: Change for other method
    if( USE_OPTIMIZATION ):
        (_,_,_,TLd) = compute_vector_TL_restrictions_body(
            atleast_1d(t),
            B_FdMd.reshape(-1,1,6,1),
            R.reshape(-1,1,3,3))

    state.TLd = TLd
    I_TLd = R @ TLd
    state.I_TLd = I_TLd
    
    (qd) = q_desired_direction(state)
    state.qd = qd
    
    # Auxiliar variable - mu is projection of mud on current cable directions
    mud = qd @ mt(qd) @ R @ TLd
    state.mud = mud
    
    mu = q @ mt(q) @ R @ TLd
    state.mu = mu
    
    
    # Load dynamics
    
    dp = v
    dR = R @ skew(o)

    # Ideal actuation (actuation on "cable" level)
    
    dv = Fd / mL + g*e3
    do = inv(I) @ Md -  inv(I) @ skew(o)@I@o    
    
    # Real actuation - Actuation is total force at quadrotor position
    
    F = sum(mu, axis=-3, keepdims=True)
    M = sum(skew(rho).reshape(-1,n_cables,n_dims,3) @ mt(R) @ mu, axis=-3, keepdims=True)
    
    state.F = F
    state.M = M
    
    dv = F / mL + g*e3
    do = inv(I) @ M -  inv(I) @ skew(o)@I@o
    
    state.dv = dv
    state.do = do
    
    (Fd,dFd) = F_feedback(state)
    (Md,dMd) = M_feedback(state)
    state.dFd = dFd
    state.dMd = dMd
    
    dB_Fd = mt(R @ skew(o)) @ Fd + mt(R) @ dFd   
    dB_FdMd = concatenate([dB_Fd,dMd],axis=-2)
    state.dB_FdMd = dB_FdMd 

    dTLd_min = (P.T @ inv(P @ P.T) @ dB_FdMd).reshape(shape(delta_TLd))
    state.dTLd_min = dTLd_min 
    
    (_, dTLd) = cable_tensions(state)
    state.dTLd = dTLd    
    dI_TLd = R @ skew(o) @ TLd + R @ dTLd
    state.dI_TLd = dI_TLd
    
    (_,dqd,oqd) = q_desired_direction(state)
    state.dqd = dqd
    state.oqd = oqd
    
    dq = skew(oq) @ q
    state.dq = dq
    
    # Auxiliar variable - mu is projection of mud on current cable directions
    dmud = (dqd @ mt(qd) @ R @ TLd
            + qd @ mt(dqd) @ R @ TLd
            + qd @ mt(qd) @ R @ skew(o) @ TLd
            + qd @ mt(qd) @ R @ dTLd)
    state.dmud = dmud
    
    dmu = (dq @ mt(q) @ R @ TLd
          + q @ mt(dq) @ R @ TLd
          + q @ mt(q) @ R @ skew(o) @ TLd
          + q @ mt(q) @ R @ dTLd)
    state.dmu = dmu
       
    dF = sum(dmu, axis=-3, keepdims=True)
    dM = sum(skew(rho).reshape(-1,n_cables,n_dims,3) @ (mt(R @ skew(o)) @ mu + mt(R) @ dmu), axis=-3, keepdims=True)

    state.dF = dF
    state.dM = dM
    
    d2v = dF / mL
    d2o = inv(I) @ dM -  inv(I) @ skew(do)@I@o -  inv(I) @ skew(o)@I@do

    state.d2v = d2v
    state.d2o = d2o
    
    (Fd,dFd,d2Fd) = F_feedback(state)
    (Md,dMd,d2Md) = M_feedback(state)
    state.d2Fd = d2Fd
    state.d2Md = d2Md
    
    d2B_Fd = mt(R @ skew(o) @ skew(o) + R @ skew(do)) @ Fd + mt(R @ skew(o)) @ dFd + mt(R @ skew(o)) @ dFd + mt(R) @ d2Fd   
    
    d2B_FdMd = concatenate([d2B_Fd,d2Md],axis=-2)
    state.d2B_FdMd = d2B_FdMd 
    
    # TODO
    d2delta_TLd = 0*ddelta_TLd
    state.d2delta_TLd = d2delta_TLd
    
    d2TLd_min = (P.T @ inv(P @ P.T) @ d2B_FdMd).reshape(shape(delta_TLd))
    state.d2TLd_min = d2TLd_min 
    (_,_,d2TLd) = cable_tensions(state)
    state.d2TLd = d2TLd
    d2I_TLd = (R @ skew(o) @ skew(o) @ TLd 
               + R @ skew(do) @ TLd 
               + R @ skew(o) @ dTLd
               + R @ skew(o) @ dTLd
               + R @ d2TLd)
    state.d2I_TLd = d2I_TLd
    (_,_,_,d2qd,doqd) = q_desired_direction(state)
    state.d2qd = d2qd
    state.doqd = doqd
    
#     print("Fd: ", Fd.shape)
#     print("dFd: ", dFd.shape)
#     print("d2Fd: ", d2Fd.shape)
    
#     print("TLd: ", TLd.shape)
#     print("dTLd: ", dTLd.shape)
#     print("d2TLd: ", d2TLd.shape)
    
    state.dp = dp
    state.dv = dv
    
    state.dR = dR
    state.do = do
    
    # Cable dynamics
    
    (q_actuation, e_q, e_oq) = q_feedback(q,oq,qd,oqd,doqd)
    
    state.q_actuation = q_actuation
    state.e_q = e_q
    state.e_oq = e_oq
    
    a = dv - g*e3 + R @ skew(o)@skew(o) @ rho - R @ skew(rho) @ do
    
    u_paralel = mu + mQ * lq * norm(oq, axis=-2, keepdims=True)**2 * q + mQ * q @ mt(q) @ a
    
#     print("u_paralel: ", u_paralel.shape)
    
#     print("q: ", q.shape)
#     print("q_actuation: ", q_actuation.shape)
#     print("a: ", a.shape)
    
    # Actuation without cross-terms
    u_perp = mQ * lq * skew(q) @ q_actuation - mQ * skew(q)@skew(q) @ a 

#     print("u_perp: ", u_perp.shape)
    
    pd = pr(t)
    dpd = dpr(t)
    d2pd = d2pr(t)
    d3pd = d3pr(t)

    pe = p-pd
    ve = v-dpd
    ae = dv-d2pd

    Rd = Rr(t)
    od = omegar(t)
    taud = taur(t)
    dtaud = dtaur(t)
    d2taud = d2taur(t)

    e_R = 1/2 * unskew(mt(Rd) @ R - mt(R) @ Rd)
    e_o = (o - od)
    de_o = do - taud
    # d2e_o = dtau - dtaud

    zoq = (
        + oq
        + skew(q) @ skew(q) @ oqd
        + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(qd) @ R @ TLd)
        - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ TLd) 
        + kq / L_q * mt(skew(q)) @ qd
    )
    
    # TODO / FIXME
    if(REMOVE_INSTABILITY_TERMS):
       zoq = (
        + oq
        + skew(q) @ skew(q) @ oqd
        + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(qd) @ R @ TLd)
        - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ TLd) 
        + kq / L_q * mt(skew(q)) @ qd
    )
    
    state.zoq = zoq
    
    u_perp_d =  - (mQ * lq) * skew(q) @ (
        + 1 / lq * skew(q) @ a 
        + skew(dq) @ skew(q) @ oqd
        + skew(q) @ skew(dq) @ oqd
        + skew(q) @ skew(q) @ doqd
        + 1 / (L_q*mL) * skew(dq) @ (X_pv @ pe + ve) * (mt(qd) @ R @ TLd)
        + 1 / (L_q*mL) * skew(q) @ (X_pv @ ve + ae) * (mt(qd) @ R @ TLd)
        + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(dqd) @ R @ TLd)
        + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(qd) @ R @ skew(o) @ TLd)
        + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(qd) @ R @ dTLd)
        - L_a / L_q * skew(dq) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ TLd) 
        - L_a / L_q * skew(q) @ R @ skew(o) @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ TLd) 
        - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ de_o * (mt(qd) @ R @ TLd) 
        - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(dqd) @ R @ TLd) 
        - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ skew(o) @ TLd) 
        - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ dTLd) 
        + kq / L_q * mt(skew(dq)) @ qd
        + kq / L_q * mt(skew(q)) @ dqd
        + koq / L_oq * (zoq)
        - L_q / L_oq * skew(q) @ qd
    )
    
    # TODO / FIXME
    if(REMOVE_INSTABILITY_TERMS):
        u_perp_d =  - (mQ * lq) * skew(q) @ (
            + 1 / lq * skew(q) @ a 
            + skew(dq) @ skew(q) @ oqd
            + skew(q) @ skew(dq) @ oqd
            + skew(q) @ skew(q) @ doqd
            + 1 / (L_q*mL) * skew(dq) @ (X_pv @ pe + ve) * (mt(qd) @ R @ TLd)
            + 1 / (L_q*mL) * skew(q) @ (X_pv @ ve + ae) * (mt(qd) @ R @ TLd)
            + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(dqd) @ R @ TLd)
            + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(qd) @ R @ skew(o) @ TLd)
            + 1 / (L_q*mL) * skew(q) @ (X_pv @ pe + ve) * (mt(qd) @ R @ dTLd)
            - L_a / L_q * skew(dq) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ TLd) 
            - L_a / L_q * skew(q) @ R @ skew(o) @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ TLd) 
            - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ de_o * (mt(qd) @ R @ TLd) 
            - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(dqd) @ R @ TLd) 
            - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ skew(o) @ TLd) 
            - L_a / L_q * skew(q) @ R @ skew(rho).reshape(-1,n_cables,n_dims,3) @ inv(I) @ e_o * (mt(qd) @ R @ dTLd) 
            + kq * mt(skew(dq)) @ qd
            + kq * mt(skew(q)) @ dqd
            + koq / L_oq * (zoq)
            - L_q / L_oq * skew(q) @ qd
        )
        
#     print("u_perp_d: ", u_perp_d.shape)
    
    # Actuation with cross-terms
    if not NO_U_PERP_CROSS_TERMS:
        u_perp = u_perp_d
    
    
    state.u_perp_d = u_perp_d
    
    state.a = a
    state.u_paralel = u_paralel
    state.u_perp = u_perp

    u = u_perp + u_paralel
    state.u = u
    
    doq = 1 / lq * skew(q) @ a - 1 / (mQ * lq) * skew(q) @ u_perp

    state.doq = doq
    
    # Quadrotor dynamics
    
#     print("t: ", t.shape)
#     print("u_paralel: ", u_paralel.shape)
#     print("u_perp: ", u_perp.shape)
#     print("u: ", u.shape)
    
    # DEBUG: Independent Rd control
    (qRd,qod,qtaud) = qR_desired(t,u)
    
#     print("qR: ", qR.shape)
#     print("qo: ", qo.shape)
#     print("qRd: ", qRd.shape)
#     print("qod: ", qod.shape)
#     print("qtaud: ", qtaud.shape)
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
#     qT = norm(qFd, axis=-3, keepdims=True)
    
    state.qFd = qFd
    state.qT = qT
    state.qF = qT * - r3
    
    # Real actuation - Quadrotor dynamics included

    mu_real = q @ mt(q) @ (state.qF + mQ * g * e3)
    state.mu_real = mu_real
    
    F_real = sum(mu_real, axis=-3, keepdims=True)
    M_real = sum(skew(rho).reshape(-1,n_cables,n_dims,3) @ mt(R) @ mu_real, axis=-3, keepdims=True)

    state.F_real = F_real
    state.M_real = M_real
    
    if(SIMULATE_REAL_QUADROTOR):
        dv = F_real / mL + g*e3
        state.dv = dv
        
        do = inv(I) @ M_real -  inv(I) @ skew(o)@I@o
        state.do = do
        
        a_real = dv - g*e3 + R @ skew(o)@skew(o) @ rho - R @ skew(rho) @ do
        state.a_real = a_real
        
        doq_real = 1 / lq * skew(q) @ a_real - 1 / (mQ * lq) * skew(q) @ state.qF
        state.doq = doq_real

    # Lyapunov verification
    
    de_q = cross(dqd,q,axis=-2) + cross(qd,dq,axis=-2)
    state.de_q = de_q
    
    dV = (
        - mt(pe) @ Kp @ X_pv @ (pe) 
        - mt(pe) @ Kv @ X_pv @ (ve)
        - mt(ve) @ (Kv - X_pv) @ ve 
        - L_a * ko * mt( e_o ) @ ( e_o )
        - kq * sum( mt(qd) @ skew(q) @ mt(skew(q)) @ qd , axis=-3, keepdims=True)
        - koq * sum( mt( zoq ) @ ( zoq ), axis=-3, keepdims=True)
        )

    state.dV = dV
    
    # Output time derivative as 1-d array and not a 2D nx1 array
    
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
    
#     print("dp =", dp.ravel())
#     print("dv =", dv.ravel())
#     print("dR =", dR.ravel())
#     print("do =", do.ravel())
#     print("dq =", dq.ravel())
#     print("doq =", doq.ravel())
#     print("--------------------")
    
    if atleast_1d(t % 1 < 1e-3).all():
        print(t)

    return dy, state


# ### Use optimization to solve for TL

# In[26]:


# Find TL solution with 1) minimum norm and 2) that observes the angle restrictions 

TL_len = n_cables*n_dims

def compute_vector_TL_restrictions_body(t,FdMd,R):
    
    (P,G) = matrix_nullspace()
    
    B_TL_min = (P.T @ inv(P @ P.T) @ FdMd).reshape(-1,n_cables,n_dims,1)
    
    TL_min = empty_like(B_TL_min)
    TL_opt = empty_like(B_TL_min)
    B_TL_opt = empty_like(B_TL_min)
    
    TL_min = R @ B_TL_min
    
    
    for i, ti in enumerate(t):
        Ri = R[i,0,...]
        FdMdi = FdMd[i,0,...]
        
        R_mult = kron( eye(n_cables) , Ri)
        TL_mini = R_mult @ B_TL_min[i].reshape(n_cables*n_dims,1)
               
        ### Setup optimization
        
        Q = eye(TL_len)

        # minimum angle with inertial/body z-axis
        theta = theta_min_for_optimization

        A = (R_mult @ P.T).T
        b = FdMdi.ravel()

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
        
        v_N = Ri @ [ cos(theta),            0,   sin(theta)] # E
        v_E = Ri @ [          0,   cos(theta),   sin(theta)] # N
        v_S = Ri @ [-cos(theta),            0,   sin(theta)] # W
        v_W = Ri @ [          0,  -cos(theta),   sin(theta)] # S
        
        # Match "rho" in SCORE lab
        Ai = vstack([
                concatenate([v_N , zeros(9)]),
                concatenate([v_E, zeros(9)]),
                concatenate([zeros(3) , v_S, zeros(6)]),
                concatenate([zeros(3) , v_E , zeros(6)]),
                concatenate([zeros(6) , v_S, zeros(3)]),
                concatenate([zeros(6) , v_W, zeros(3)]),
                concatenate([zeros(9) , v_N]),
                concatenate([zeros(9) , v_W]),
                ])

        ##   Comment to choose -> theta angle with inertial z-axis
        ## Uncomment to choose -> theta angle with body z-axis
        # Ai = (R_mult.T @ Ai.T).T

        bi = zeros(Ai.shape[0])
        
        ### Run optimization

        # use solution from previous timestep
        try:
            x0 = compute_vector_TL_restrictions_body.x0
        except AttributeError:
            x0 = random.randn(TL_len)

        def loss(x, sign=1.):
            return sign * (0.5 * x.T @ Q @ x)

        def jac(x, sign=1.):
            return sign * (x.T @ Q)

        cons = ({'type':'eq',
                'fun':lambda x: A @ x - b,
                 'jac':lambda x: A},
               {'type':'ineq',
                'fun':lambda x: Ai @ x - bi,
                 'jac':lambda x: Ai}
               )

        options = {'disp':False}

        # CVXOPT
        try:
            sol=solvers.qp(
                            P=matrix(Q, tc='d'),
                            q=matrix(np.zeros(len(x0)),tc='d'), 
                            A=matrix(A, tc='d'), 
                            b=matrix(b, tc='d'),
                            G=matrix(-Ai, tc='d'), 
                            h=matrix(-bi, tc='d'),
                            initvals=matrix(x0, tc='d'),
                            options={'show_progress':False}
                          )
            x = np.array(sol['x'])
            TL_opt[i] = x.reshape(n_cables,n_dims,1)
            compute_TL.x0 = x
        except Exception as e:
            print('CVXopt did not found a solution')
            print(e)
            return
        pass
    
#         # scipy.optimize

#         res = optimize.minimize(loss, 
#                                 x0, 
#                                 jac=jac,
#                                 constraints=cons,
#                                 options=options,
#                                 tol=1e-9)

#         if(res.success):
# #             print(res)
#             TL_opt[i] = res.x.reshape(n_cables,3,1)
#             compute_vector_TL_restrictions_body.x0 = res.x
#         else:
#             print(f't = {ti:0.3} \t No solution:', res.message)
#             raise RuntimeError('No solution found for optimization problem!') 
# #             TL_opt[i] = zeros_like(res.x)
        
        B_TL_opt[i] = (Ri.T @ TL_opt[i])
        
#         print(B_TL_opt)
    
    return (TL_min, TL_opt, B_TL_min, B_TL_opt)



def compute_vector_TL_restrictions_inertial(t,FdMd,R):
    
    (P,G) = matrix_nullspace()
    
    B_TL_min = (P.T @ inv(P @ P.T) @ FdMd).reshape(-1,n_cables,n_dims,1)
    
    TL_min = empty_like(B_TL_min)
    TL_opt = empty_like(B_TL_min)
    B_TL_opt = empty_like(B_TL_min)
    
    TL_min = R @ B_TL_min
       
    for i, ti in enumerate(t):
        Ri = R[i,0,...]
        FdMdi = FdMd[i,0,...]
    
        R_mult = kron( eye(n_cables) , Ri)
        TL_mini = R_mult @ B_TL_min[i].reshape(n_cables*n_dims,1)
        
        ### Setup optimization
        
        Q = eye(TL_len)

        # minimum angle with inertial/body z-axis
        theta = theta_min_for_optimization

        A = (R_mult @ P.T).T
        b = FdMdi.ravel()

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
        
        v = rotationMatrixToEulerAngles(Ri)
        v[0] = 0
        v[1] = 0
        Ryaw = eulerAnglesToRotationMatrix(v)

        v_N = Ryaw @ [ cos(theta),            0,   sin(theta)] # E
        v_E = Ryaw @ [          0,   cos(theta),   sin(theta)] # N
        v_S = Ryaw @ [-cos(theta),            0,   sin(theta)] # W
        v_W = Ryaw @ [          0,  -cos(theta),   sin(theta)] # S
        
        # Match "rho" in SCORE lab
        Ai = vstack([
                concatenate([v_N , zeros(9)]),
                concatenate([v_E, zeros(9)]),
                concatenate([zeros(3) , v_S, zeros(6)]),
                concatenate([zeros(3) , v_E , zeros(6)]),
                concatenate([zeros(6) , v_S, zeros(3)]),
                concatenate([zeros(6) , v_W, zeros(3)]),
                concatenate([zeros(9) , v_N]),
                concatenate([zeros(9) , v_W]),
                ])

        ##   Comment to choose -> theta angle with inertial z-axis
        ## Uncomment to choose -> theta angle with body z-axis
        # Ai = (R_mult.T @ Ai.T).T

        bi = zeros(Ai.shape[0])
        
        ### Run optimization

        # todo: use solution from previous timestep
        if(i>1):
            x0 = TL_opt[i-1].ravel()
        else:
            x0 = random.randn(TL_len)

        def loss(x, sign=1.):
            return sign * (0.5 * x.T @ Q @ x)

        def jac(x, sign=1.):
            return sign * (x.T @ Q)

        cons = ({'type':'eq',
                'fun':lambda x: A @ x - b,
                 'jac':lambda x: A},
               {'type':'ineq',
                'fun':lambda x: Ai @ x - bi,
                 'jac':lambda x: Ai}
               )

        options = {'disp':False}
        
        res = optimize.minimize(loss, 
                                x0, 
                                jac=jac,
                                constraints=cons,
                                options=options,
                                tol=1e-9)

        if(res.success):
            TL_opt[i] = res.x.reshape(n_cables,3,1)
        else:
#             print(f't = {ti:0.3} \t No solution:', res.message)
            raise RuntimeError('No solution found for optimization problem!')
#             TL_opt[i] = zeros_like(res.x)
        
        B_TL_opt[i] = (Ri.T @ TL_opt[i])
    
    return (TL_min, TL_opt, B_TL_min, B_TL_opt)


# ### Initial states

# In[27]:


# Quadrotor initial states

p0 = zeros((3,1))

v0 = zeros((3,1))

# R0 = expm(skew(zeros((3,1))))
R0 = Rx(10*pi/180) @ Ry(5*pi/180)

o0 = zeros((3,1))

delta_TLd0 = zeros((12,1))

state0 = SimpleNamespace()

state0.p = p0
state0.v = v0
state0.pd = pr(t0)
state0.dpd = dpr(t0)
state0.d2pd = d2pr(t0)

state0.R = R0
state0.o = o0
state0.Rd = Rr(t0)
state0.od = omegar(t0)
state0.taud = taur(t0)

# # Initialize delta_TL0 at feasible location (outside of potential singularity)
B_Fd0 = R0.T @ F_feedback(state0)
Md0 = M_feedback(state0)
B_FdMd0 = vstack([B_Fd0,Md0])

# (TLd_min0, TLd_opt0, B_TLd_min0, B_TLd_opt0) = compute_vector_TL_restrictions_inertial(
#     [t0],
#     B_FdMd0.reshape(-1,1,n_cables,1),
#     R0.reshape(-1,1,3,3))

# delta_TLd0 = (B_TLd_opt0 - B_TLd_min0).reshape(-1,n_cables,n_dims,1)


# # TODO: choose q0
# d = ones([3,1]) / sqrt(3)
# q0 = tile(d,n_cables).T.reshape(n_cables,n_dims,1)

# q0 = tile(e3,n_cables).T.reshape(n_cables,n_dims,1)

# Strings point outwards of center of mass
theta0 = deg2rad(10)
q0 = array([  [[-sin(theta0)/sqrt(2)], [-sin(theta0)/sqrt(2)], [cos(theta0)]],
              [[ sin(theta0)/sqrt(2)], [-sin(theta0)/sqrt(2)], [cos(theta0)]],
              [[ sin(theta0)/sqrt(2)], [ sin(theta0)/sqrt(2)], [cos(theta0)]],
              [[-sin(theta0)/sqrt(2)], [ sin(theta0)/sqrt(2)], [cos(theta0)]] ])

oq0 = zeros((n_cables, n_dims,1))

# qR0 = tile(eye(3),n_cables).T.reshape(n_cables,3,3)
qR0 = tile(Rx(0*pi/180),n_cables).T.reshape(n_cables,3,3)

qo0 = zeros((n_cables, n_dims,1))

V0 = array(0)


# ### Auxiliar functions for packing and unpacking the system state

# In[28]:


def pack_state(p,v,R,o,delta_TLd,q,oq,qR,qo,V):
#     print(p.shape)
#     print(v.shape)
#     print(R.reshape(-1,1).shape)
#     print(o.shape)
#     print(delta_TLd.reshape(-1,1).shape)
    return vstack([p.reshape(-1,1),
                   v.reshape(-1,1),
                   R.reshape(-1,1),
                   o.reshape(-1,1),
                   delta_TLd.reshape(-1,1),
                   q.reshape(-1,1),
                   oq.reshape(-1,1),
                   qR.reshape(-1,1),
                   qo.reshape(-1,1),
                   V.reshape(-1,1),])

state_sizes = [3,      # p
               3,      # v
               3*3,    # R
               3,      # o
               n_dims*n_cables,   # TLd
               n_dims*n_cables,   # q
               n_dims*n_cables,   # oq
               3*3*n_cables,    # quadrotor R
               3*n_cables,   # quadrotor o
               1,   # V
              ]
state_idx = cumsum(state_sizes)

def unpack_state(x):
    (p,v,R,o,delta_TLd,q,oq,qR,qo,V,_) = split(x,state_idx)
    
    return (p.reshape(3,1),
            v.reshape(3,1),
            R.reshape(3,3),
            o.reshape(3,1),
            delta_TLd.reshape(n_cables,3,1),
            q.reshape(n_cables,3,1),
            oq.reshape(n_cables,3,1),
            qR.reshape(n_cables,3,3),
            qo.reshape(n_cables,3,1),
            V)


# In[29]:


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

# In[30]:


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


# In[31]:


# from functools import partial

# # Define fixed arguments for 'f' used for ODE 
# f_ode = partial(f,args=args)
f_ode = f


# ### Auxiliar function to unpack a complete solution by states

# In[32]:


def unpack_solution(x):
    
#     import pdb; pdb.set_trace()
    
    try:
        (p,v,R,o,delta_TLd,q,oq,qR,qo,V,_) = split(x,state_idx,axis=-1)
    except IndexError:
        (p,v,R,o,delta_TLd,q,oq,qR,qo,V,_) = split(x,state_idx)
        
    return (p.reshape(-1,1,3,1),
            v.reshape(-1,1,3,1),
            R.reshape(-1,1,3,3),
            o.reshape(-1,1,3,1),
            delta_TLd.reshape(-1,n_cables,3,1),
            q.reshape(-1,n_cables,3,1),
            oq.reshape(-1,n_cables,3,1),
            qR.reshape(-1,n_cables,3,3),
            qo.reshape(-1,n_cables,3,1),
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


# In[33]:


# # TEST - Unpack(Pack(x)) = x

# p00 = p0.reshape(-1,1,3,1)
# v00 = v0.reshape(-1,1,3,1)
# R00 = R0.reshape(-1,1,3,3)
# o00 = o0.reshape(-1,1,3,1)
# delta_TLd00 = delta_TLd0.reshape(-1,n_cables,3,1)
# q00 = q0.reshape(-1,n_cables,3,1)
# oq00 = oq0.reshape(-1,n_cables,3,1)
# qR00 = qR0.reshape(-1,n_cables,3,3)
# qo00 = qo0.reshape(-1,n_cables,3,1)
# V00 = V0.reshape(-1,1,1,1)

# orig = (p00,v00,R00,o00,delta_TLd00,q00,oq00,qR00,qo00,V00)
# state0 = pack_solution(p00,v00,R00,o00,delta_TLd00,q00,oq00,qR00,qo00,V00)
# (p00,v00,R00,o00,delta_TLd00,q00,oq00,qR00,qo00,V00) = unpack_solution(mt(state0))
# (sum(abs(orig[0] - p00)) + 
# sum(abs(orig[1] - v00)) + 
# sum(abs(orig[2] - R00)) + 
# sum(abs(orig[3] - o00)) +
# sum(abs(orig[4] - delta_TLd00)) +
# sum(abs(orig[5] - q00)) +
# sum(abs(orig[6] - oq00)) +
# sum(abs(orig[7] - qR00)) +
# sum(abs(orig[8] - qo00)) +
# sum(abs(orig[9] - V00)))


# # stop here for .py  #
