#!/usr/bin/env python3
# coding: utf-8

# From https://github.com/mavlink/mavros/blob/master/mavros/src/lib/ftf_frame_conversions.cpp
import numpy as np
from scipy.spatial.transform import Rotation

# Used for Rotation matrix conversions

# R_ENU_NED_R = Rotation.from_euler('xyz',[np.pi, 0.0, np.pi/2])
NED_ENU_R = Rotation.from_dcm(np.array([[0.0,  1.0,  0.0],
                                           [ 1.0, 0.0, 0.0],
                                           [ 0.0,  0.0, -1.0]]))
# AIRCRAFT_BASELINK_R = Rotation.from_euler('xyz',[0.0, 0.0, np.pi])
AIRCRAFT_BASELINK_R = Rotation.from_dcm(np.array([[1.0, 0.0,  0.0],
                                               [0.0, -1.0, 0.0],
                                               [0.0,  0.0, -1.0]]))
# Used for position conversions

NED_ENU_REFLECTION_XY = np.array([[0.0, 1.0,  0.0],
                                   [1.0, 0.0, 0.0],
                                   [ 0.0,  0.0, 1.0]])
NED_ENU_REFLECTION_Z = np.array([[1.0, 0.0,  0.0],
                                   [0.0, 1.0, 0.0],
                                   [ 0.0,  0.0, -1.0]])

def transform_orientation_I(R):
    return NED_ENU_R*R

def transform_orientation_B(R):
    return R*AIRCRAFT_BASELINK_R

def transform_position_I(p):
    return NED_ENU_REFLECTION_XY @ (NED_ENU_REFLECTION_Z @ p)

def transform_position_B(p):
    return AIRCRAFT_BASELINK_R.apply(p)

def transform_omega_B(w):
    return AIRCRAFT_BASELINK_R.apply(w)