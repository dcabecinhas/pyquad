#!/usr/bin/env python3

# POINT_MASS = True
# from point_mass_4_vehicles import skew, mt, pack_state, process_state, n_cables, n_dims, mQ, mL, g, state_sizes, rho, sim_gains, sim_parameters

POINT_MASS = False
from rigid_body_4_vehicles import skew, mt, pack_state, process_state, n_cables, n_dims, mQ, mL, g, state_sizes, rho, sim_gains, sim_parameters

import rospy
import math

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import norm

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, WrenchStamped, Quaternion, Vector3
from std_msgs.msg import Header
from mavros_msgs.msg import AttitudeTarget

from frame_conversions import transform_orientation_I, transform_orientation_B, transform_position_I, transform_position_B, transform_omega_B

from threading import Thread

import time
from time import sleep

from vehicle import Vehicle

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
np.set_printoptions(floatmode="fixed")
np.set_printoptions(sign=" ")

def F_vector_to_Rotation(F):
    F = F.reshape(-1,3,1)
    e3 = np.array([0,0,1]).reshape(-1,3,1)

    # Angle of
    angle = (np.arccos(mt(-e3) @ F / norm(F, axis=-2, keepdims = True)))
    vector = np.cross(-e3,F,axis=-2)
    vector = vector / norm(vector, axis=-2, keepdims = True)

    return Rotation.from_rotvec((angle * vector).reshape(-1,3))

# def F_vector_to_Rotation(F):
#     F = F.reshape(-1,n_cables,3,1)
#     e3 = np.array([0,0,1]).reshape(-1,1,3,1)

#     # Angle of
#     angle = (np.arccos(mt(-e3) @ F / norm(F, axis=-2, keepdims = True)))
#     vector = np.cross(-e3,F,axis=-2)
#     vector = vector / norm(vector, axis=-2, keepdims = True)

# #     print(Rotation.from_rotvec((angle * vector).reshape(-1,3)))

#     return Rotation.from_rotvec((angle * vector).reshape(-1,3))




class ROSLoadController():

    def __init__(self, number_of_vehicles = 1, vehicle_prefix = 'iris_', cable_prefix = 'cable_', load_prefix = 'load', mavros_prefix = 'uav_' ):

        self.log = np.zeros([1+sum(state_sizes),10000])

        # super(ROSLoadController, self).__init__()
        self.n = number_of_vehicles

        self.pL = np.zeros([1,3,1])
        self.vL = np.zeros([1,3,1])
        self.RL = np.zeros([1,3,3])
        self.oL = np.zeros([1,3,1])
        self.pQ = np.zeros([number_of_vehicles,3,1])
        self.vQ = np.zeros([number_of_vehicles,3,1])
        self.RQ = np.zeros([number_of_vehicles,3,3])
        self.oQ = np.zeros([number_of_vehicles,3,1])

        self.position_active = True
        self.attitude_active = False
        self.load_active = False

        self.q_input = np.zeros([self.n,4,1])
        self.T_input = np.zeros([self.n,1,1])

        self.sub_topics_ready = {
            key: False
            # for key in ['vehicle_odom', 'vehicle_force', 'cable_odom', 'cable_force',
            # 'load_odom', 'load_force' ]
            for key in ['vehicle_odom', 'load_odom', 'vehicle_force', 'load_force' ]
        }

        # ROS subscribers
        for i in range(number_of_vehicles):
            self.vehicle_odom_sub = rospy.Subscriber(vehicle_prefix + str(i) + '/odom',
                                                Odometry,
                                                self.vehicle_odom_callback,
                                                callback_args=i)
            self.vehicle_force_sub = rospy.Subscriber(vehicle_prefix + str(i) + '/force',
                                                WrenchStamped,
                                                self.vehicle_force_callback,
                                                callback_args=i)
            # self.cable_odom_sub = rospy.Subscriber(cable_prefix + str(i) + '/odom',
            #                                     Odometry,
            #                                     self.local_position_callback,
            #                                     callback_args=i)
            # self.cable_force_sub = rospy.Subscriber(cable_prefix + str(i) + '/force',
            #                                     WrenchStamped,
            #                                     self.local_position_callback,
            #                                     callback_args=i)

        self.load_odom_sub = rospy.Subscriber(load_prefix + '/odom',
                                            Odometry,
                                            self.load_odom_callback)
        self.load_force_sub = rospy.Subscriber(load_prefix + '/force',
                                            WrenchStamped,
                                            self.load_force_callback)

        # ROS Publishers

        # Attitude
        self.att = AttitudeTarget()
        # self.att_setpoint_pub = [rospy.Publisher(
        #         f'load_controller/vehicle_{i}/attitude', AttitudeTarget, queue_size=1) for i in range(number_of_vehicles)]
        self.att_setpoint_pub = [rospy.Publisher(
                mavros_prefix + str(i) + '/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1) for i in range(number_of_vehicles)]

        # send setpoints in seperate thread to better prevent failsafe
        self.att_thread = Thread(target=self.send_att, args=())
        self.att_thread.daemon = True
        self.att_thread.start()

        # Pose
        self.pos = PoseStamped()
        # self.pos_setpoint_pub = [rospy.Publisher(
        #         f'load_controller/vehicle_{i}/position', PoseStamped, queue_size=1) for i in range(number_of_vehicles)]
        self.pos_setpoint_pub = [rospy.Publisher(
                mavros_prefix + str(i) + '/mavros/setpoint_position/local', PoseStamped, queue_size=1) for i in range(number_of_vehicles)]

        # send setpoints in separate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos, args=())
        self.pos_thread.daemon = True
        self.pos_thread.start()

    #
    # Callback functions
    #
    def load_odom_callback(self, data):
        p_enu = np.array([data.pose.pose.position.x,
                        data.pose.pose.position.y,
                        data.pose.pose.position.z])
        p_ned = transform_position_I(p_enu)

        v_enu = np.array([data.twist.twist.linear.x,
                        data.twist.twist.linear.y,
                        data.twist.twist.linear.z])
        v_ned = transform_position_I(v_enu)

        q_enu = np.array([data.pose.pose.orientation.x,
                        data.pose.pose.orientation.y,
                        data.pose.pose.orientation.z,
                        data.pose.pose.orientation.w])
        q_ned = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q_enu))).as_quat()
        R_ned = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q_enu))).as_dcm()

        o_enu = np.array([data.twist.twist.angular.x,
                        data.twist.twist.angular.y,
                        data.twist.twist.angular.z])
        o_ned = transform_omega_B(o_enu)

        self.pL = p_ned.reshape([1,3,1])
        self.vL = v_ned.reshape([1,3,1])
        self.RL = R_ned.reshape([1,3,3])
        self.oL = o_ned.reshape([1,3,1])

        if not self.sub_topics_ready['load_odom']:
            self.sub_topics_ready['load_odom'] = True

    def vehicle_odom_callback(self, data, vehicle_number):
        p_enu = np.array([data.pose.pose.position.x,
                        data.pose.pose.position.y,
                        data.pose.pose.position.z])
        p_ned = transform_position_I(p_enu)

        v_enu = np.array([data.twist.twist.linear.x,
                        data.twist.twist.linear.y,
                        data.twist.twist.linear.z])
        v_ned = transform_position_I(v_enu)

        q_enu = np.array([data.pose.pose.orientation.x,
                        data.pose.pose.orientation.y,
                        data.pose.pose.orientation.z,
                        data.pose.pose.orientation.w])
        q_ned = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q_enu))).as_quat()
        R_ned = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q_enu))).as_dcm()

        o_enu = np.array([data.twist.twist.angular.x,
                        data.twist.twist.angular.y,
                        data.twist.twist.angular.z])
        o_ned = transform_omega_B(o_enu)

        self.pQ[vehicle_number] = p_ned.reshape([3,1])
        self.vQ[vehicle_number] = v_ned.reshape([3,1])
        self.RQ[vehicle_number] = R_ned.reshape([3,3])
        self.oQ[vehicle_number] = o_ned.reshape([3,1])

        if not self.sub_topics_ready['vehicle_odom']:
            self.sub_topics_ready['vehicle_odom'] = True

        # If quadrotors are upside-down then terminate simulation
        e3 = np.array([0,0,1]).reshape(-1,3,1)


    def vehicle_force_callback(self, data, vehicle_number):
        # self.local_position = data

        if not self.sub_topics_ready['vehicle_force']:
            self.sub_topics_ready['vehicle_force'] = True

    def load_force_callback(self, data):
        # self.local_position = data

        if not self.sub_topics_ready['load_force']:
            self.sub_topics_ready['load_force'] = True

    def wait_for_topics(self, timeout):
        """wait for simulation to be ready, make sure we're getting topic info
        from all topics by checking dictionary of flag values set in callbacks,
        timeout(int): seconds"""
        rospy.loginfo("waiting for subscribed topics to be ready")
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        simulation_ready = False
        for i in range(timeout * loop_freq):
            if all([value for value in self.sub_topics_ready.values()]):
                simulation_ready = True
                rospy.loginfo("simulation topics ready | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue(simulation_ready, (
            "failed to hear from all subscribed simulation topics | topic ready flags: {0} | timeout(seconds): {1}".
            format(self.sub_topics_ready, timeout)))


    # Publish target attitude

    def send_att(self):
        try:
            rate = rospy.Rate(50)  # Hz
            self.att.body_rate = Vector3()
            self.att.header = Header()
            self.att.header.frame_id = "base_footprint"

            self.att.orientation = Quaternion(*self.q_input[0])
            self.att.thrust = self.T_input[0]

            self.att.type_mask = 7  # ignore body rate
        except Exception as e:
            print(e)
            pass

        while not rospy.is_shutdown():
            # print(f"-----")
            # print(f"a = {rospy.get_time()}")
            self.att.header.stamp = rospy.Time.now()

            if(not self.position_active):
                self.computeSystemStates()
                if(self.attitude_active):
                    (T,q) = self.controller_attitude()
                if(self.load_active):
                    (T,q) = self.controller_load()
                    # print(f"q_input = {q}")
                    # print(f"T_input = {T}")

                    # print(f"  q = {q}")
                    # print(f"  T = {T}")
                    # print(f"  T.shape = {T.shape}")

                # print(f"b = {rospy.get_time()}")

                for i in range(self.n):
                    q_ENU = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q))).as_quat()

                    # print(f"q_input_ENU = {q_ENU}")
                    # print(f"T_input = {T}")
                    # print(f"q_quad_ENU = {self.gazebo_q_quad_ENU}")

                    self.att.orientation = Quaternion(*q_ENU[i])
                    self.att.thrust = T[i]

                    self.att_setpoint_pub[i].publish(self.att)

                    # print(f"c_{i} = {rospy.get_time()}")

                # print(f"d = {rospy.get_time()}")

                # print(f"Published att. at t={rospy.get_time()}")
                # if(self.load_active):
                #     # print(f'Published to UAV {i}:')
                #     # print(f"  q[i] = {q_ENU[i]}")
                #     # print(f"  T[i] = {T[i]}")

                #     # Debug
                #     print("Published attitude commands: ")

                #     # Debug
                #     e3 = np.array([0,0,1]).reshape(-1,3,1)

                #     i = 0

                #     print(f"   T_{i}: ", T[i])
                #     print(f"   q_{i}: ", q[i])
                #     print(f"   q_{i} ENU: ", q_ENU[i])
                #     print(f"   euler_{i}: ", Rotation.from_quat(q[i]).as_euler('zyx'))
                #     print(f"   euler_{i} NED: ", Rotation.from_quat(q_ENU[i]).as_euler('zyx'))
                #     print(f" r3_{i}    : ", (Rotation.from_quat(q[i]).as_dcm() @ -e3).ravel())
                #     print(f" r3_{i} NED: ", (Rotation.from_quat(q_ENU[i]).as_dcm() @ e3).ravel())
                #     print("---")

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    # Publish target position

    def send_pos(self):
        rate = rospy.Rate(5)  # Hz
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()

            # while True:
            t = rospy.get_time()

            self.pd = np.array([0,0,-0.5]).reshape([3,1])
            self.Dpd = np.array([0,0,0]).reshape([3,1])
            self.D2pd = np.array([0,0,0]).reshape([3,1])

            if(self.position_active):
                pd = self.controller_position()

                pd_ENU = transform_position_I(pd)

                # set a position setpoint
                self.pos.pose.position.x = pd_ENU[0]
                self.pos.pose.position.y = pd_ENU[1]
                self.pos.pose.position.z = pd_ENU[2]

                # set yaw angle
                yaw_degrees = 0
                yaw = math.radians(yaw_degrees)
                quaternion = transform_orientation_I(transform_orientation_B((
                    Rotation.from_euler('zyx', [yaw,0,0])
                    ))).as_quat()
                self.pos.pose.orientation = Quaternion(*quaternion)

                for i in range(self.n):
                    self.pos_setpoint_pub[i].publish(self.pos)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def assertTrue(self, result, text):
        if(not result):
            rospy.loginfo(text)

    def computeSystemStates(self):
        # Convertions from ROS to MAVLINK frames (ENU to NED)

        # In https://github.com/mavlink/mavros/issues/216
        # body-fixed NED → ROS ENU: (x y z)→(x -y -z) or (w x y z)→(x -y -z w)
        # local NED → ROS ENU: (x y z)→(y x -z) or (w x y z)→(y x -z w)

        # See https://github.com/mavlink/mavros/blob/master/mavros/src/lib/ftf_frame_conversions.cpp
        # static const auto NED_ENU_Q = quaternion_from_rpy(M_PI, 0.0, M_PI_2);

        # compute auxiliar states
        self.pL_RB = self.pL + self.RL @ rho
        self.vL_RB = self.vL + self.RL @ skew(self.oL) @ rho

        self.q = (self.pL_RB - self.pQ) / norm(self.pL_RB - self.pQ, axis=-2, keepdims=True)
        self.dq = - skew(self.q) @ skew(self.q) @ (self.vL_RB - self.vQ) / norm(self.pL_RB - self.pQ, axis=-2, keepdims=True)
        self.o = skew(self.q) @ self.dq

        # print('pL = ', self.pL)

        # print('pQ = ', self.pQ)

        # print('q = ', self.q)

        # print('dq = ', self.dq)

        # print('o = ', self.o)

        return

    def controller_load(self):

        self.delta_TLd = np.zeros(n_cables*n_dims)
        self.V = np.zeros([1,1])

        t = rospy.get_time() # - self.t0
        dt = t - self.t_prev
        self.t_prev = t
        # print(f"==========    t = {t:.3} / dt = {dt:.3}   ==========")
        # print(f"self.pL = {self.pL.ravel()}")
        # print(f"self.vL = {self.vL.ravel()}")
        # print(f"self.pQ = {self.pQ.ravel()}")
        # print(f"self.vQ = {self.vQ.ravel()}")
        # print(f"self.RL = {self.RL.ravel()}")
        # print(f"self.oL = {self.oL.ravel()}")
        # print(f"self.delta_TLd = {self.delta_TLd.ravel()}")
        # print(f"self.q  = {self.q.ravel()}")
        # print(f"self.o  = {self.o.ravel()}")
        # print(f"self.RQ = {self.RQ.ravel()}")
        # print(f"self.oQ = {self.oQ.ravel()}")
        # print(f"self.qQ = {self.qQ.ravel()}")
        # e3 = np.array([0,0,1]).reshape([3,1])
        # r3 = self.RQ @ e3
        # print(f"r3Q     = {r3.ravel()}")
        # print(f"oQ      = {self.oQ.ravel()}")
        # print("===================================")

        y = pack_state(self.pL,
                    self.vL,
                    self.RL,
                    self.oL,
                    self.delta_TLd,
                    self.q,
                    self.o,
                    self.RQ,
                    self.oQ,
                    self.V)

        try:
            dy, state = process_state(t,y.T)
            # print(f"state.pd = {state.pd}")
            # print(f"state.Fd = {state.Fd}")
            # print(f"state.q = {state.q}")
            # print(f"state.qd = {state.qd}")
        except Exception as e:
            print("Couldn't compute state")
            print(e)
            return

        state.u = state.u.reshape(-1,3,1)

        # T = np.zeros([self.n,1,1])
        # q = np.zeros([self.n,4,1])

        r3_d = (- state.u / norm(state.u, axis=-2, keepdims=True)).reshape(-1,3,1)
        qd = state.qd.reshape(-1,3,1)
        R = F_vector_to_Rotation(state.u)

        kT = 0.5

        q = R.as_quat().reshape(-1,4,1)

        T = self.SI2RC(norm(state.u,axis=-2,keepdims=True))

        # print("u = ", state.u)
        # print("T = ", T)

        # print(f"state.u.shape = {state.u.shape}")
        # print(f"R.shape = {R.shape}")
        # print(f"q.shape = {q.shape}")
        # print(f"T.shape = {T.shape}")

        T = np.clip(T,0,1)

        e3 = np.array([0,0,1]).reshape(-1,3,1)
        r3 = R.as_dcm() @ e3
        yrp = R.as_euler('zyx')
        yaw = yrp[...,0]

        yawd = np.deg2rad(0)

        R_yaw = Rotation.from_rotvec(np.squeeze(r3 * (yawd-yaw).reshape(4,1,1)))
        R_yaw_0 = (R_yaw * R)

        q = R_yaw_0.as_quat()


        if ( (self.RQ[:,2,2] < 0 ).any() or (self.q[:,2,0] < 0 ).any()):
            filename = time.strftime("%Y%m%d_%H%M%S")
            directory = "numpy_logs"

            import os
            try:
                os.mkdir( directory )
            except FileExistsError:
                pass
            np.savez_compressed(os.path.join(directory,filename),
            # np.savez_compressed(filename),
                state_vector=load_controller.log,
                sim_parameters=sim_parameters,
                sim_gains=sim_gains
                )

            print("Quadrotors crashed!... SHUTTING DOWN ROS.")
            # import sys
            # sys.exit()
            rospy.signal_shutdown("Quadrotors crashed.")

            sleep(1)

            # Kill whole program from thread - https://stackoverflow.com/questions/1489669/how-to-exit-the-entire-application-from-a-python-thread
            import os
            os._exit(1)

        # print("u = ", state.u[0])
        # print("q[0] = ",q[0])
        # print("Euler (qF)[0]:")
        # print(R_yaw_0[0].as_euler('zyx'))

        # print('u = ', state.u)
        # print('T = ', T)
        # print('q = ', q)

        # print("Fd = ", state.Fd)
        # print("Md = ", state.Md)
        # print("R = ", state.R)
        # print("Rd = ", state.Rd)
        # print("q = ", state.q)
        # print("qd = ", state.qd)

        if POINT_MASS:
            # Point mass
            state_vector = np.vstack([state.p.reshape(-1,1),
                    state.v.reshape(-1,1),
                    np.eye(3).reshape(-1,1),
                    np.zeros(3).reshape(-1,1),
                    state.delta_TLd.reshape(-1,1),
                    state.q.reshape(-1,1),
                    state.oq.reshape(-1,1),
                    state.qR.reshape(-1,1),
                    state.qo.reshape(-1,1),
                    state.V.reshape(-1,1),])
        else:
            # Rigid body
            state_vector = np.vstack([state.p.reshape(-1,1),
                    state.v.reshape(-1,1),
                    state.R.reshape(-1,1),
                    state.o.reshape(-1,1),
                    state.delta_TLd.reshape(-1,1),
                    state.q.reshape(-1,1),
                    state.oq.reshape(-1,1),
                    state.qR.reshape(-1,1),
                    state.qo.reshape(-1,1),
                    state.V.reshape(-1,1),])

        self.log[:-1,self.k] = state_vector.ravel()
        self.log[-1,self.k] = t
        self.k = self.k + 1

        # print(f"Computed att. at t={t}")

#         print(f"t_ROS = {rospy.get_time()},   t = {t}")
#         print(f"NED commands: ")
#         print(f"   T = {T}")
#         print(f"   q = {q}")
#         print( "---")

        return (T, q)


    def SI2RC(self,F):
        T = 0.0167*F + 0.3351
        return T

    def controller_attitude(self):

        e3 = np.array([0,0,1]).reshape([3,1])

        kp = 1.5 * np.array([1,1,1]).reshape([3,1])
        kv = 3 * np.array([1,1,1]).reshape([3,1])

        # Reference

        self.pd = np.array([[
            [ 1.2/2,  0.8/2, -3.0],
            [-1.2/2,  0.8/2, -3.0],
            [-1.2/2, -0.8/2, -3.0],
            [ 1.2/2, -0.8/2, -3.0]
            ]]).reshape(-1,3,1)

        # Start diagonal for eight maneuver
        self.pd = np.array([[
            [ 1.2/2 ,  0    , -3.0],
            [   0   ,  0.8  , -3.0],
            [-1.2/2 ,  0    , -3.0],
            [   0   ,  0.8  , -3.0]
            ]]).reshape(-1,3,1)
        self.Dpd = 0*self.pd
        self.D2pd = 0*self.pd

        # print('self.pQ = ', self.pQ)
        # print('self.pd = ', self.pd)

        # PD controller for position

        Fd = - kp*(self.pQ - self.pd) - kv*(self.vQ - self.Dpd) + mQ*self.D2pd - (mQ + mL/4)*g*e3

        R = F_vector_to_Rotation(Fd)

        # print(f"R * norm(Fd) * (-e3)  = {(R).as_dcm() @ (-e3) * norm(state.Fd)}")

        # q = R.as_quat().reshape(-1,4,1)

        # kT = 2.5
        # T = kT*(norm(Fd, axis=-2, keepdims=True)/(mQ*g) - 1.0) + 0.5

        T = self.SI2RC(norm(Fd,axis=-2,keepdims=True))

        T = np.clip(T,0,1)

        yaw = np.deg2rad(0)
        R_yaw = Rotation.from_rotvec(yaw*np.array([0,0,1]))

        q = (R*R_yaw).as_quat()

        # # For thrust identification
        # print("T   = ", T.ravel() )
        # print("p_z =", self.pQ[:,2,:].ravel() )

        return (T, q)


    def controller_position(self):
        return self.pd


if __name__ == '__main__':

    number_of_vehicles = 4

    print("Initializing node...")
    rospy.init_node('pyquad', anonymous=True)

    print("Initializing Controller...")
    load_controller = ROSLoadController(number_of_vehicles=4)

    print("Waiting for topics...")
    load_controller.wait_for_topics(30)

    load_controller.computeSystemStates()

    print("* Attitude control *")
    load_controller.position_active = False
    load_controller.attitude_active = True
    load_controller.load_active = False

    # Arm quadrotors and set mode to offboard
    quad_list = []
    for i in range(number_of_vehicles):
        print(f"[{i}] Start vehicle object...")
        quad = Vehicle(ID=i)

        # print(i, '- a')
        quad.wait_for_topics(10)
        # print(i, '- b')
        quad.thread.start()
        # print(i, '- c')
        quad_list.append(quad)

    rospy.sleep(7.0)

    print("* Load control *")
    load_controller.t0 = rospy.get_time()
    load_controller.k = 0
    load_controller.t_prev = -0.01
    load_controller.position_active = False
    load_controller.attitude_active = False
    load_controller.load_active = True

    sleep(60)

    filename = time.strftime("%Y%m%d_%H%M%S")
    directory = "numpy_logs"

    import os
    try:
        os.mkdir( directory )
    except FileExistsError:
        pass
    np.savez_compressed(os.path.join(directory,filename),
        state_vector=load_controller.log,
        sim_parameters=sim_parameters,
        sim_gains=sim_gains
        )
    
    import os
    os._exit(1)

    # while True:

    #     print("Now listening for options (1,2,3) - 0 to exit:")
    #     try:
    #         c = input()[0]
    #     except:
    #         continue

    #     if(c=='1'):
    #         print("* Position control *")
    #         load_controller.position_active = True
    #         load_controller.attitude_active = False
    #         load_controller.load_active = False

    #     if(c=='2'):
    #         print("* Attitude control *")
    #         load_controller.position_active = False
    #         load_controller.attitude_active = True
    #         load_controller.load_active = False

    #     if(c=='3'):
    #         print("* Load control *")
    #         load_controller.t0 = rospy.get_time()
    #         load_controller.k = 0
    #         load_controller.t_prev = -0.01
    #         load_controller.position_active = False
    #         load_controller.attitude_active = False
    #         load_controller.load_active = True

    #     if(c=='0'):
    #         filename = time.strftime("%Y%m%d_%H%M%S")
    #         directory = "numpy_logs"

    #         import os
    #         try:
    #             os.mkdir( directory )
    #         except FileExistsError:
    #             pass
    #         np.savez_compressed(os.path.join(directory,filename),
    #             state_vector=load_controller.log,
    #             sim_parameters=sim_parameters,
    #             sim_gains=sim_gains
    #             )

    #         print("SHUTTING DOWN ROS.")
    #         # import sys
    #         # sys.exit()
    #         rospy.signal_shutdown("Finished.")
    #         break
