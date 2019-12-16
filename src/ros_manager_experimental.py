#!/usr/bin/env python3

import os
import yaml
import yappi

configuration_geometry = os.environ["GEOMETRY_CONFIGURATION"]

with open(configuration_geometry) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    cfg = yaml.load(file)
    
if cfg['POINT_MASS']:
    from point_mass_controller import skew, mt, pack_state, process_state, n_cables, n_dims, mQ, mL, g, state_sizes, rho, pr
else:
    from rigid_body_controller import skew, mt, pack_state, process_state, n_cables, n_dims, mQ, mL, g, state_sizes, rho, pr

import rospy
import math

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import norm

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, WrenchStamped, Quaternion, Vector3
from std_msgs.msg import Header
from mavros_msgs.msg import AttitudeTarget, State, RCIn

from frame_conversions import transform_orientation_I, transform_orientation_B, transform_position_I, transform_position_B, transform_omega_B

from threading import Thread

import time
from time import sleep

from vehicle import Vehicle

from pymavlink import mavutil

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

class ROSLoadController():

    def __init__(self, cfg = []):

        # Log is state_size + pd + load active + vehicle armed + first letter of vehicle mode + time
        self.log = np.zeros([7+sum(state_sizes),10000])

        # super(ROSLoadController, self).__init__()
        self.n = cfg['number_of_vehicles']

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
        self.q_input[:,0,0] = 1
        self.T_input = np.zeros([self.n,1,1])

        self.state = State()

        self.rc = RCIn()

        self.sub_topics_ready = {
            key: False
            for key in ['vehicle_odom', 'load_odom', 'state', 'rc']
        }

        self.mode = number_of_vehicles*['.']
        self.armed = np.zeros([self.n,1,1])

        # ROS subscribers
        for i in range(number_of_vehicles):
            self.vehicle_odom_sub = rospy.Subscriber(cfg[i]['vicon'] + '/odom',
                                                Odometry,
                                                self.vehicle_odom_callback,
                                                callback_args=i)
            self.state_sub = rospy.Subscriber(cfg[i]['mavros'] + '/mavros/state', 
                                                State,
                                                self.state_callback,
                                                callback_args=i)

        self.rc_sub = rospy.Subscriber(cfg[0]['mavros'] + '/mavros/rc/in',
                                                RCIn,
                                                self.rc_callback)

        print(f"Subscribing to {cfg[0]['mavros'] + '/mavros/rc/in'}")

        self.load_odom_sub = rospy.Subscriber(cfg['load']['vicon'] + '/odom',
                                            Odometry,
                                            self.load_odom_callback)

        # ROS Publishers

        # Attitude

        self.att = AttitudeTarget()

        self.att_setpoint_pub = [rospy.Publisher(
                cfg[i]['mavros'] + '/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1) for i in range(number_of_vehicles)]
        print('publish attitude at: ', cfg[i]['mavros'] + '/mavros/setpoint_raw/attitude')

        # send setpoints in seperate thread to better prevent failsafe
        self.att_thread = Thread(target=self.send_att, args=())
        self.att_thread.daemon = True
        self.att_thread.start()

        # Pose
        self.pos = PoseStamped()
        self.pos_setpoint_pub = [rospy.Publisher(
                cfg[i]['mavros']  + '/mavros/setpoint_position/local', PoseStamped, queue_size=1) for i in range(number_of_vehicles)]

        # send setpoints in separate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos, args=())
        self.pos_thread.daemon = True
        # self.pos_thread.start()

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
        # q_ned = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q_enu))).as_quat()
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
        # q_ned = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q_enu))).as_quat()
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


    def state_callback(self, data, vehicle_number):
        if self.state.armed != data.armed:
            rospy.loginfo(f"Vehicle {vehicle_number}: armed state changed from {self.state.armed} to {data.armed}")

        if self.state.connected != data.connected:
            rospy.loginfo(f"Vehicle {vehicle_number}: connected changed from {self.state.connected} to {data.connected}")

        if self.state.mode != data.mode:
            rospy.loginfo(f"Vehicle {vehicle_number}: mode changed from {self.state.mode} to {data.mode}")

        if self.state.system_status != data.system_status:
            rospy.loginfo(f"Vehicle {vehicle_number}: system_status changed from {mavutil.mavlink.enums['MAV_STATE'][self.state.system_status].name} to {mavutil.mavlink.enums['MAV_STATE'][data.system_status].name}")

        self.mode = data.mode
        self.armed = data.armed
        
        self.state = data
    
        # print(f"Vehicle {vehicle_number} MODE: ",self.state.mode)

        # mavros publishes a disconnected state message on init
        if not self.sub_topics_ready['state'] and data.connected:
            self.sub_topics_ready['state'] = True

    def rc_callback(self, data):
            self.rc = data.channels

            if(self.rc[6] < 1300):
                if(not self.load_active):
                    # If first commutation into load controller
                    self.t0 = rospy.get_time()
                self.load_active = True
            else:
                self.load_active = False
            
            if not self.sub_topics_ready['rc']:
                self.sub_topics_ready['rc'] = True

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

            # print('self.att.target_system:', self.att.target_system)
            # print('self.att.target_component:', self.att.target_component)

        except Exception as e:
            print(e)
            pass

        while not rospy.is_shutdown():
            self.att.header.stamp = rospy.Time.now()

            if(not self.position_active):
                self.computeSystemStates()
                if(self.attitude_active):
                    (T,q) = self.controller_attitude()
                if(self.load_active):
                    (T,q) = self.controller_load()

                for i in range(self.n):
                    q_ENU = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q))).as_quat()

                    self.att.orientation = Quaternion(*q_ENU[i])
                    self.att.thrust = T[i]

                    try:
                        self.att_setpoint_pub[i].publish(self.att)
                    except rospy.exceptions.ROSException as e:
                        print(f"Vehicle {i} - Attitude: ", e)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
                # time.sleep(0.001)
            except rospy.ROSInterruptException:
                pass

    # Publish target position

    def send_pos(self):
        rate = rospy.Rate(10)  # Hz
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()

            # while True:
            t = rospy.get_time()

            self.pd = np.array([cfg[i]['starting_position'] for i in range(cfg['number_of_vehicles'])]).reshape(cfg['number_of_vehicles'],cfg['dimensions'],1)
            self.Dpd = 0*self.pd
            self.D2pd = 0*self.pd

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
                    try:
                        self.pos_setpoint_pub[i].publish(self.pos)
                    except rospy.exceptions.ROSException as e:
                        print(e)
                    

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
        
        return

    def controller_load(self):

        self.delta_TLd = np.zeros(n_cables*n_dims)
        self.V = np.zeros([1,1])

        t = rospy.get_time() - self.t0
        dt = t - self.t_prev
        self.t_prev = t
        
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
        except Exception as e:
            print("Couldn't compute state")
            print(e)
            return

        state.u = state.u.reshape(-1,3,1)

        state.u = state.u

        # r3_d = (- state.u / norm(state.u, axis=-2, keepdims=True)).reshape(-1,3,1)
        # qd = state.qd.reshape(-1,3,1)
        
        R = F_vector_to_Rotation(state.u)

        q = R.as_quat().reshape(-1,number_of_vehicles,1)

        T = self.SI2RC(norm(state.u,axis=-2,keepdims=True))

        T = np.clip(T,0,1)

        e3 = np.array([0,0,1]).reshape(-1,3,1)
        r3 = R.as_dcm() @ e3
        yrp = R.as_euler('zyx')
        yaw = yrp[...,0]

        yawd = np.deg2rad(0)

        R_yaw = Rotation.from_rotvec(np.squeeze(r3 * (yawd-yaw).reshape(number_of_vehicles,1,1)))
        R_yaw_0 = (R_yaw * R)

        q = R_yaw_0.as_quat()

        # print("q: ", mt(self.q))
        # print("RQ: ", mt(self.RQ))

        # if ( (self.RQ[:,2,2] < 0 ).any() or (self.q[:,2,0] < 0 ).any()):
        if ( (self.RQ[:,2,2] < 0 ).any()):    
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
                # sim_parameters=sim_parameters,
                # sim_gains=sim_gains
                configuration=cfg
                )

            print("Quadrotors crashed!... SHUTTING DOWN ROS.")
            # import sys
            # sys.exit()
            rospy.signal_shutdown("Quadrotors crashed.")

            sleep(1)

            # Kill whole program from thread - https://stackoverflow.com/questions/1489669/how-to-exit-the-entire-application-from-a-python-thread
            import os
            os._exit(1)

        if cfg['POINT_MASS']:
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

        self.log[:-7,self.k] = state_vector.ravel()
        self.log[-7:-4, self.k] = pr(t)
        self.log[-4,self.k] = self.load_active
        self.log[-3,self.k] = self.armed
        self.log[-2,self.k] = ord(self.mode[0])
        self.log[-1,self.k] = t
        self.k = self.k + 1

        # print(f"Computed att. at t={t}")

        # print(f"t_ROS = {rospy.get_time()},   t = {t}")
        # print(f"NED commands: ")
        # print(f"   q = {q}")
        # print( "---")
        # print(f"Load   --- ")
        # print(f"Fd: {mt(state.Fd)}")
        # print(f"Fq: {mt(state.u)}")
        # print(f"q: {mt(state.q)}")
        # print(f"qd: {mt(state.qd)}")
        # print(f"T: {T.ravel()}")
        # euler = Rotation.from_quat(q).as_euler('zyx')
        # print(f"Euler: {euler}")
        # print(f"pd: {state.pd}")
        # print('=========================================')
        # print(state)
        # print(f"t = {t}")

        return (T, q)


    def SI2RC(self,F):
        # # For simulation
        # T = 0.0167*F + 0.3351
        # # For Intel Aero experimental
        F = F*0.92
        T = F / 20 - 0.05
        return T

    def controller_attitude(self):

        e3 = np.array([0,0,1]).reshape([3,1])

        kp = np.array([4.0,4.0,2.0]).reshape([3,1])
        kv = np.array([4.0,4.0,1.5]).reshape([3,1])

        self.pd = np.array([cfg[i]['starting_position'] for i in range(cfg['number_of_vehicles'])]).reshape(cfg['number_of_vehicles'],cfg['dimensions'],1)
        self.Dpd = 0*self.pd
        self.D2pd = 0*self.pd

        # PD controller for position

        # Fd = - kp*(self.pQ - self.pd) - kv*(self.vQ - self.Dpd) + mQ*self.D2pd - (mQ + mL/number_of_vehicles)*g*e3

        Fd = - kp*(self.pQ - self.pd) - kv*(self.vQ - self.Dpd) + mQ*self.D2pd - mQ*g*e3

        # FIXME: Necessary because of bad thrust identification
        Fd = Fd * 1.075

        R = F_vector_to_Rotation(Fd)

        T = self.SI2RC(norm(Fd,axis=-2,keepdims=True))

        T = np.clip(T,0,1)

        yaw = np.deg2rad(0)
        R_yaw = Rotation.from_rotvec(yaw*np.array([0,0,1]))

        q = (R*R_yaw).as_quat()

        # print('pd shape = ', self.pd.shape)
        # print('pQ shape = ', self.pQ.shape)
        # print('T shape = ', T.shape)
        # for i in range(cfg['number_of_vehicles']):
        #     print(f"Thrust   ---    T: {T[i]},   pd = {self.pd[i].T},   p = {self.pQ[i].T}")
        #     print("Force      ---   Fd: ")
        #     print(np.column_stack([Fd[i].ravel(), (- kp*(self.pQ - self.pd))[i].ravel(), (- kv*(self.vQ - self.Dpd))[i].ravel()]))

        return (T, q)


    def controller_position(self):
        return self.pd


if __name__ == '__main__':

    yappi.start()

    print("Initializing node...")
    rospy.init_node('pyquad', anonymous=True)

    number_of_vehicles = cfg['number_of_vehicles']

    print(f'=== number_of_vehicles ==={number_of_vehicles}')

    print("Initializing Controller...")
    load_controller = ROSLoadController(cfg)

    print(f'=== number_of_vehicles ==={number_of_vehicles}')

    print("Waiting for topics...")
    load_controller.wait_for_topics(5)

    load_controller.computeSystemStates()

    print("* Attitude control *")
    load_controller.position_active = False
    load_controller.attitude_active = True
    load_controller.load_active = False

    print(f'=== number_of_vehicles === {number_of_vehicles}')
    
    load_controller.t0 = rospy.get_time()
    load_controller.k = 0
    load_controller.t_prev = -0.01

    # Arm quadrotors and set mode to offboard
    quad_list = []
    for i in range(number_of_vehicles):
        print(f"[{i}] Start vehicle object...")
        quad = Vehicle(ID=i, vehicle_cfg=cfg[i])

        # quad.wait_for_topics(10)
        # quad.thread.start()
        quad_list.append(quad)

    # rospy.sleep(150.0)

    # print("* Load control *")
    # load_controller.t0 = rospy.get_time()
    # load_controller.k = 0
    # load_controller.t_prev = -0.01
    # load_controller.position_active = False
    # load_controller.attitude_active = False
    # load_controller.load_active = True

    while True:

        print("Now listening for options (1,2,3) - 0 to exit:")
        try:
            c = input()[0]
        except:
            continue

        # # if(c=='1'):
        # #     print("* Position control *")
        # #     load_controller.position_active = True
        # #     load_controller.attitude_active = False
        # #     load_controller.load_active = False

        # if(c=='1'):
        #     print("* Attitude control *")
        #     load_controller.position_active = False
        #     load_controller.attitude_active = True
        #     load_controller.load_active = False

        # if(c=='2'):
        #     print("* Load control *")
        #     load_controller.t0 = rospy.get_time()
        #     load_controller.k = 0
        #     load_controller.t_prev = -0.01
        #     load_controller.position_active = False
        #     load_controller.attitude_active = False
        #     load_controller.load_active = True

        if(c=='0'):

            # import sys
            # sys.exit()
            rospy.signal_shutdown("Finished.")
            break
    
    print("Saving logs")


    filename = time.strftime("%Y%m%d_%H%M%S")
    directory = "numpy_logs"

    import os
    try:
        os.mkdir( directory )
    except FileExistsError:
        pass
    np.savez_compressed(os.path.join(directory,filename),
        state_vector=load_controller.log,
        configuration=cfg
        )
    
    # Profiling results
    # yappi.get_func_stats().print_all()
    # yappi.get_thread_stats().print_all()

    from datetime import datetime
    func_stats = yappi.get_func_stats()
    func_stats.save('callgrind.out.' + datetime.now().isoformat(), 'CALLGRIND')
    yappi.stop()
    yappi.clear_stats()

    print("SHUTTING DOWN ROS.")

    import os
    os._exit(1)
