#!/usr/bin/env python3

from single_quadrotor import *

import rospy
import math

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import norm

from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Twist, Quaternion, Vector3
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
from mavros_msgs.msg import ActuatorControl, AttitudeTarget, Altitude, \
                            ExtendedState, HomePosition, ManualControl, \
                            PositionTarget, State, StatusText, Thrust, \
                            WaypointList
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, \
                            WaypointPush

from mavros.utils import *

from pymavlink import mavutil

from frame_conversions import *

from std_msgs.msg import Header
from threading import Thread

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(floatmode="fixed")
np.set_printoptions(sign=" ")

def F_vector_to_Rotation(F):
    F = F.ravel()
    e3 = np.array([0,0,1])

    # Angle of 
    angle = (arccos((-e3) @ F / norm(F)))
    vector = cross(-e3,F)
    vector = vector / norm(vector)

    return Rotation.from_rotvec(angle * vector)
    

class MavrosQuad():

    def __init__(self, *args):
        super(MavrosQuad, self).__init__(*args)

        self.altitude = Altitude()
        self.extended_state = ExtendedState()
        self.global_position = NavSatFix()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.mission_wp = WaypointList()
        self.state = State()
        self.mav_type = None

        self.local_velocity = TwistStamped()

        self.gazebo_load_name = 'rigid_body_load_1_vehicle::rb_link'
        self.gazebo_load_pose = Pose()
        self.gazebo_load_twist = Twist()

        self.gazebo_quad_name = 'rigid_body_load_1_vehicle::base_link_0'
        self.gazebo_quad_pose = Pose()
        self.gazebo_quad_twist = Twist()

        self.gazebo_imu_name = 'rigid_body_load_1_vehicle::iris_0/imu_link'
        self.gazebo_imu_pose = Pose()
        self.gazebo_imu_twist = Twist()

        self.q_input = np.array([0,0,0,1])
        self.T_input = 0

        self.offboard_position_active = True
        self.offboard_attitude_active = False
        self.offboard_load_active = False

        self.sub_topics_ready = {
            key: False
            for key in [
                'alt', 'ext_state', 'gazebo', 'global_pos', 'home_pos',
                'local_pos', 'local_vel', 'mission_wp', 'state', 'imu'
            ]
        }

        # ROS services
        service_timeout = 30
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.wait_for_service('mavros/param/get', service_timeout)
            rospy.wait_for_service('mavros/cmd/arming', service_timeout)
            rospy.wait_for_service('mavros/mission/push', service_timeout)
            rospy.wait_for_service('mavros/mission/clear', service_timeout)
            rospy.wait_for_service('mavros/set_mode', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            rospy.logerr("failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy('mavros/param/get', ParamGet)
        self.set_arming_srv = rospy.ServiceProxy('mavros/cmd/arming',
                                                 CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)
        self.wp_clear_srv = rospy.ServiceProxy('mavros/mission/clear',
                                               WaypointClear)
        self.wp_push_srv = rospy.ServiceProxy('mavros/mission/push',
                                              WaypointPush)

        # ROS subscribers
        self.alt_sub = rospy.Subscriber('mavros/altitude', Altitude,
                                        self.altitude_callback)
        self.ext_state_sub = rospy.Subscriber('mavros/extended_state',
                                              ExtendedState,
                                              self.extended_state_callback)
        self.global_pos_sub = rospy.Subscriber('mavros/global_position/global',
                                               NavSatFix,
                                               self.global_position_callback)
        self.home_pos_sub = rospy.Subscriber('mavros/home_position/home',
                                             HomePosition,
                                             self.home_position_callback)
        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose',
                                              PoseStamped,
                                              self.local_position_callback)
        self.mission_wp_sub = rospy.Subscriber('mavros/mission/waypoints', 
                                            WaypointList, 
                                            self.mission_wp_callback)
        self.state_sub = rospy.Subscriber('mavros/state', 
                                            State,
                                            self.state_callback)
        
        self.local_vel_sub = rospy.Subscriber('mavros/local_position/velocity_local',
                                            TwistStamped,
                                            self.local_velocity_callback)
        
        self.imu_sub = rospy.Subscriber('/mavros/imu/data',
                                            Imu,
                                            self.imu_callback)
                                            
        self.gazebo_sub = rospy.Subscriber('/gazebo/link_states',
                                            LinkStates,
                                            self.gazebo_callback)

        # ROS Publishers

        # Attitude
        self.att = AttitudeTarget()
        self.att_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)

        # send setpoints in seperate thread to better prevent failsafe
        self.att_thread = Thread(target=self.send_att, args=())
        self.att_thread.daemon = True
        self.att_thread.start()

        # Pose

        self.pos = PoseStamped()
        self.pos_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_position/local', PoseStamped, queue_size=1)

        # send setpoints in seperate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos, args=())
        self.pos_thread.daemon = True
        self.pos_thread.start()

    #
    # Callback functions
    #
    def altitude_callback(self, data):
        self.altitude = data

        # amsl has been observed to be nan while other fields are valid
        if not self.sub_topics_ready['alt'] and not math.isnan(data.amsl):
            self.sub_topics_ready['alt'] = True

    def extended_state_callback(self, data):
        if self.extended_state.vtol_state != data.vtol_state:
            rospy.loginfo("VTOL state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_VTOL_STATE']
                [self.extended_state.vtol_state].name, mavutil.mavlink.enums[
                    'MAV_VTOL_STATE'][data.vtol_state].name))

        if self.extended_state.landed_state != data.landed_state:
            rospy.loginfo("landed state changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_LANDED_STATE']
                [self.extended_state.landed_state].name, mavutil.mavlink.enums[
                    'MAV_LANDED_STATE'][data.landed_state].name))

        self.extended_state = data

        if not self.sub_topics_ready['ext_state']:
            self.sub_topics_ready['ext_state'] = True

    def global_position_callback(self, data):
        self.global_position = data

        if not self.sub_topics_ready['global_pos']:
            self.sub_topics_ready['global_pos'] = True

    def home_position_callback(self, data):
        self.home_position = data

        if not self.sub_topics_ready['home_pos']:
            self.sub_topics_ready['home_pos'] = True

    def local_position_callback(self, data):
        self.local_position = data

        if not self.sub_topics_ready['local_pos']:
            self.sub_topics_ready['local_pos'] = True

    def mission_wp_callback(self, data):
        if self.mission_wp.current_seq != data.current_seq:
            rospy.loginfo("current mission waypoint sequence updated: {0}".
                          format(data.current_seq))

        self.mission_wp = data

        if not self.sub_topics_ready['mission_wp']:
            self.sub_topics_ready['mission_wp'] = True

    def state_callback(self, data):
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(
                self.state.armed, data.armed))

        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(
                self.state.connected, data.connected))

        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(
                self.state.mode, data.mode))

        if self.state.system_status != data.system_status:
            rospy.loginfo("system_status changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_STATE'][
                    self.state.system_status].name, mavutil.mavlink.enums[
                        'MAV_STATE'][data.system_status].name))

        self.state = data

        # mavros publishes a disconnected state message on init
        if not self.sub_topics_ready['state'] and data.connected:
            self.sub_topics_ready['state'] = True

    def local_velocity_callback(self, data):
        self.local_velocity = data

        if not self.sub_topics_ready['local_vel']:
            self.sub_topics_ready['local_vel'] = True
    
    def imu_callback(self, data):
        self.imu_orientation = data.orientation
        self.imu_angular_velocity = data.angular_velocity
        
        if not self.sub_topics_ready['imu']:
                self.sub_topics_ready['imu'] = True

    def gazebo_callback(self, data):
        ind = data.name.index(self.gazebo_load_name)
        self.gazebo_load_pose = data.pose[ind]
        self.gazebo_load_twist = data.twist[ind]
        
        ind = data.name.index(self.gazebo_quad_name)
        self.gazebo_quad_pose = data.pose[ind]
        self.gazebo_quad_twist = data.twist[ind]

        ind = data.name.index(self.gazebo_imu_name)
        self.gazebo_imu_pose = data.pose[ind]
        self.gazebo_imu_twist = data.twist[ind]

        if not self.sub_topics_ready['gazebo']:
                self.sub_topics_ready['gazebo'] = True

    #
    # Helper methods
    #
    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in range(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue(arm_set, (
            "failed to set arm | new arm: {0}, old arm: {1} | timeout(seconds): {2}".
            format(arm, old_arm, timeout)))

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in range(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue(mode_set, (
            "failed to set mode | new mode: {0}, old mode: {1} | timeout(seconds): {2}".
            format(mode, old_mode, timeout)))

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

    def wait_for_landed_state(self, desired_landed_state, timeout, index):
        rospy.loginfo("waiting for landed state | state: {0}, index: {1}".
                      format(mavutil.mavlink.enums['MAV_LANDED_STATE'][
                          desired_landed_state].name, index))
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        landed_state_confirmed = False
        for i in range(timeout * loop_freq):
            if self.extended_state.landed_state == desired_landed_state:
                landed_state_confirmed = True
                rospy.loginfo("landed state confirmed | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue(landed_state_confirmed, (
            "landed state not detected | desired: {0}, current: {1} | index: {2}, timeout(seconds): {3}".
            format(mavutil.mavlink.enums['MAV_LANDED_STATE'][
                desired_landed_state].name, mavutil.mavlink.enums[
                    'MAV_LANDED_STATE'][self.extended_state.landed_state].name,
                   index, timeout)))

    def wait_for_vtol_state(self, transition, timeout, index):
        """Wait for VTOL transition, timeout(int): seconds"""
        rospy.loginfo(
            "waiting for VTOL transition | transition: {0}, index: {1}".format(
                mavutil.mavlink.enums['MAV_VTOL_STATE'][
                    transition].name, index))
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        transitioned = False
        for i in range(timeout * loop_freq):
            if transition == self.extended_state.vtol_state:
                rospy.loginfo("transitioned | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                transitioned = True
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue(transitioned, (
            "transition not detected | desired: {0}, current: {1} | index: {2} timeout(seconds): {3}".
            format(mavutil.mavlink.enums['MAV_VTOL_STATE'][transition].name,
                   mavutil.mavlink.enums['MAV_VTOL_STATE'][
                       self.extended_state.vtol_state].name, index, timeout)))

    def clear_wps(self, timeout):
        """timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        wps_cleared = False
        for i in range(timeout * loop_freq):
            if not self.mission_wp.waypoints:
                wps_cleared = True
                rospy.loginfo("clear waypoints success | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.wp_clear_srv()
                    if not res.success:
                        rospy.logerr("failed to send waypoint clear command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue(wps_cleared, (
            "failed to clear waypoints | timeout(seconds): {0}".format(timeout)
        ))

    def send_wps(self, waypoints, timeout):
        """waypoints, timeout(int): seconds"""
        rospy.loginfo("sending mission waypoints")
        if self.mission_wp.waypoints:
            rospy.loginfo("FCU already has mission waypoints")

        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        wps_sent = False
        wps_verified = False
        for i in range(timeout * loop_freq):
            if not wps_sent:
                try:
                    res = self.wp_push_srv(start_index=0, waypoints=waypoints)
                    wps_sent = res.success
                    if wps_sent:
                        rospy.loginfo("waypoints successfully transferred")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            else:
                if len(waypoints) == len(self.mission_wp.waypoints):
                    rospy.loginfo("number of waypoints transferred: {0}".
                                  format(len(waypoints)))
                    wps_verified = True

            if wps_sent and wps_verified:
                rospy.loginfo("send waypoints success | seconds: {0} of {1}".
                              format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue((
            wps_sent and wps_verified
        ), "mission could not be transferred and verified | timeout(seconds): {0}".
                        format(timeout))

    def wait_for_mav_type(self, timeout):
        """Wait for MAV_TYPE parameter, timeout(int): seconds"""
        rospy.loginfo("waiting for MAV_TYPE")
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        res = False
        for i in range(timeout * loop_freq):
            try:
                res = self.get_param_srv('MAV_TYPE')
                if res.success:
                    self.mav_type = res.value.integer
                    rospy.loginfo(
                        "MAV_TYPE received | type: {0} | seconds: {1} of {2}".
                        format(mavutil.mavlink.enums['MAV_TYPE'][self.mav_type]
                               .name, i / loop_freq, timeout))
                    break
            except rospy.ServiceException as e:
                rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)

        self.assertTrue(res.success, (
            "MAV_TYPE param get failed | timeout(seconds): {0}".format(timeout)
        ))
    
    # Publish target attitude

    def send_att(self):
        try:
            rate = rospy.Rate(50)  # Hz
            self.att.body_rate = Vector3()
            self.att.header = Header()
            self.att.header.frame_id = "base_footprint"
            
            self.att.orientation = Quaternion(*self.q_input)
            self.att.thrust = self.T_input
            
            self.att.type_mask = 7  # ignore body rate
        except Exception as e: 
            print(e)
            pass

        while not rospy.is_shutdown():
            self.att.header.stamp = rospy.Time.now()

            if(not self.offboard_position_active):
                self.computeSystemStates()
                if(self.offboard_attitude_active):
                    (T,q) = self.controller_attitude()
                if(self.offboard_load_active):
                    (T,q) = self.controller_load()
                    # print(f"q_input = {q}")
                    # print(f"T_input = {T}")

                q_ENU = transform_orientation_I(transform_orientation_B(Rotation.from_quat(q))).as_quat()

                # print(f"q_input_ENU = {q_ENU}")
                # print(f"T_input = {T}")
                # print(f"q_quad_ENU = {self.gazebo_q_quad_ENU}")


                self.att.orientation = Quaternion(*q_ENU)
                self.att.thrust = T
                
                self.att_setpoint_pub.publish(self.att)

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

            self.pd = np.array([0,0,-1]).reshape([3,1])
            self.Dpd = np.array([0,0,0]).reshape([3,1])
            self.D2pd = np.array([0,0,0]).reshape([3,1])
            
            if(self.offboard_position_active):
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

                self.pos_setpoint_pub.publish(self.pos)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass


    def log_topic_vars(self):
        """log the state of topic variables"""
        rospy.loginfo("log")
        rospy.loginfo("extended_state:\n{}".format(self.extended_state))
        rospy.loginfo("========================")
        rospy.loginfo("===== topic values =====")
        rospy.loginfo("========================")
        rospy.loginfo("altitude:\n{}".format(self.altitude))
        rospy.loginfo("========================")
        rospy.loginfo("extended_state:\n{}".format(self.extended_state))
        rospy.loginfo("========================")
        rospy.loginfo("global_position:\n{}".format(self.global_position))
        rospy.loginfo("========================")
        rospy.loginfo("home_position:\n{}".format(self.home_position))
        rospy.loginfo("========================")
        rospy.loginfo("local_position:\n{}".format(self.local_position))
        rospy.loginfo("========================")
        rospy.loginfo("mission_wp:\n{}".format(self.mission_wp))
        rospy.loginfo("========================")
        rospy.loginfo("state:\n{}".format(self.state))
        rospy.loginfo("========================")
        rospy.loginfo("local_velocity:\n{}".format(self.local_velocity))
        rospy.loginfo("========================")
        rospy.loginfo("load position:\n{}".format(self.load_pose))
        rospy.loginfo("========================")
        rospy.loginfo(f"pL: {self.pL}")
        rospy.loginfo(f"vL: {self.vL}")
        rospy.loginfo(f"qL: {self.qL}")
        rospy.loginfo(f"oL: {self.oL}")
        rospy.loginfo(f"R: {self.R}")
        rospy.loginfo(f"o: {self.o}")
        rospy.loginfo("========================")
        
    def log_states(self):
        """log the states of quadrotor-load system"""
        # rospy.loginfo("========================")
        # rospy.loginfo(f"pL: {self.pL.T}")
        # rospy.loginfo(f"vL: {self.vL.T}")
        # rospy.loginfo(f"qL: {self.qL.T}")
        # rospy.loginfo(f"oL: {self.oL.T}")
        # # rospy.loginfo(f"R: {self.R}")
        # rospy.loginfo(f"R (in RPY): {(Rotation.from_dcm(self.RQ)).as_euler('zyx')}")
        # rospy.loginfo(f"R (in quat): {(Rotation.from_dcm(self.RQ)).as_quat()}")
        # rospy.loginfo(f"o: {self.oQ.T}")
        # rospy.loginfo(f"R_imu (in RPY): {(Rotation.from_dcm(self.R_imu)).as_euler('zyx')}")
        # rospy.loginfo(f"R_imu (in quat): {(Rotation.from_dcm(self.R_imu)).as_quat()}")
        # rospy.loginfo(f"o_imu: {self.o_imu.T}")
        rospy.loginfo(f"pQ: {self.pQ.T}")
        rospy.loginfo(f"qQ: {self.qQ.T}")
        # rospy.loginfo(f"vQ: {self.vQ.T}")

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
        
        # MAVROS processing 
        
        p_quad_ENU = np.array([self.local_position.pose.position.x,
                            self.local_position.pose.position.y,
                            self.local_position.pose.position.z])

        q_quad_ENU = np.array([self.local_position.pose.orientation.x,
                                self.local_position.pose.orientation.y,
                                self.local_position.pose.orientation.z,
                                self.local_position.pose.orientation.w])

        R_quad_ENU = Rotation.from_quat(q_quad_ENU).as_dcm()

        v_quad_ENU = np.array([self.local_velocity.twist.linear.x,
                                self.local_velocity.twist.linear.y,
                                self.local_velocity.twist.linear.z])

        o_quad_ENU = np.array([self.local_velocity.twist.angular.x,
                                self.local_velocity.twist.angular.y,
                                self.local_velocity.twist.angular.z])

        # Load measurements from gazebo

        p_load_ENU = np.array([self.gazebo_load_pose.position.x,
                                self.gazebo_load_pose.position.y,
                                self.gazebo_load_pose.position.z])

        v_load_ENU = np.array([self.gazebo_load_twist.linear.x,
                                self.gazebo_load_twist.linear.y,
                                self.gazebo_load_twist.linear.z])

        # MAVROS IMU processing

        q_quad_imu_ENU = np.array([self.imu_orientation.x,
                                self.imu_orientation.y,
                                self.imu_orientation.z,
                                self.imu_orientation.w])

        R_quad_imu_ENU = Rotation.from_quat(q_quad_imu_ENU).as_dcm()

        o_quad_imu_ENU = np.array([self.imu_angular_velocity.x,
                                    self.imu_angular_velocity.y,
                                    self.imu_angular_velocity.z])

        self.R_imu = transform_orientation_I(transform_orientation_B((Rotation.from_dcm(R_quad_imu_ENU)))).as_dcm()
        self.o_imu = transform_omega_B(o_quad_imu_ENU).reshape([3,1])

        # Gazebo processing 

        ## Quadrotor base_link

        gazebo_p_quad_ENU = np.array([self.gazebo_quad_pose.position.x,
                                    self.gazebo_quad_pose.position.y,
                                    self.gazebo_quad_pose.position.z])

        gazebo_q_quad_ENU = np.array([self.gazebo_quad_pose.orientation.x,
                                    self.gazebo_quad_pose.orientation.y,
                                    self.gazebo_quad_pose.orientation.z,
                                    self.gazebo_quad_pose.orientation.w])
        self.gazebo_q_quad_ENU = gazebo_q_quad_ENU

        gazebo_R_quad_ENU = Rotation.from_quat(gazebo_q_quad_ENU).as_dcm()

        gazebo_v_quad_ENU = np.array([self.gazebo_quad_twist.linear.x,
                                    self.gazebo_quad_twist.linear.y,
                                    self.gazebo_quad_twist.linear.z])

        gazebo_o_quad_ENU = np.array([self.gazebo_quad_twist.angular.x,
                                    self.gazebo_quad_twist.angular.y,
                                    self.gazebo_quad_twist.angular.z])

        self.gazebo_pQ = transform_position_I(gazebo_p_quad_ENU.reshape([3,1]))
        self.gazebo_vQ = transform_position_I(gazebo_v_quad_ENU.reshape([3,1]))
        self.gazebo_qQ = transform_orientation_I(transform_orientation_B((Rotation.from_quat(gazebo_q_quad_ENU)))).as_quat()
        self.gazebo_oQ = transform_omega_B(gazebo_o_quad_ENU).reshape([3,1])

        ## Load link

        gazebo_p_load_ENU = np.array([self.gazebo_load_pose.position.x,
                                    self.gazebo_load_pose.position.y,
                                    self.gazebo_load_pose.position.z])

        gazebo_q_load_ENU = np.array([self.gazebo_load_pose.orientation.x,
                                    self.gazebo_load_pose.orientation.y,
                                    self.gazebo_load_pose.orientation.z,
                                    self.gazebo_load_pose.orientation.w])

        gazebo_R_load_ENU = Rotation.from_quat(gazebo_q_load_ENU).as_dcm()

        gazebo_v_load_ENU = np.array([self.gazebo_load_twist.linear.x,
                                    self.gazebo_load_twist.linear.y,
                                    self.gazebo_load_twist.linear.z])

        gazebo_o_load_ENU = np.array([self.gazebo_load_twist.angular.x,
                                    self.gazebo_load_twist.angular.y,
                                    self.gazebo_load_twist.angular.z])

        self.gazebo_pL = transform_position_I(gazebo_p_load_ENU.reshape([3,1]))
        self.gazebo_vL = transform_position_I(gazebo_v_load_ENU.reshape([3,1]))
        self.gazebo_qL = transform_orientation_I(transform_orientation_B((Rotation.from_quat(gazebo_q_load_ENU)))).as_quat()
        self.gazebo_oL = transform_omega_B(gazebo_o_load_ENU).reshape([3,1])

        ## IMU link 

        gazebo_p_imu_ENU = np.array([self.gazebo_imu_pose.position.y,
                                    self.gazebo_imu_pose.position.x,
                                    self.gazebo_imu_pose.position.z])

        gazebo_q_imu_ENU = np.array([self.gazebo_imu_pose.orientation.x,
                                    self.gazebo_imu_pose.orientation.y,
                                    self.gazebo_imu_pose.orientation.z,
                                    self.gazebo_imu_pose.orientation.w])

        gazebo_R_imu_ENU = Rotation.from_quat(gazebo_q_quad_ENU).as_dcm()

        gazebo_v_imu_ENU = np.array([self.gazebo_imu_twist.linear.x,
                                    self.gazebo_imu_twist.linear.y,
                                    self.gazebo_imu_twist.linear.z])

        gazebo_o_imu_ENU = np.array([self.gazebo_imu_twist.angular.x,
                                    self.gazebo_imu_twist.angular.y,
                                    self.gazebo_imu_twist.angular.z])

        self.gazebo_pIMU = transform_position_I(gazebo_p_imu_ENU.reshape([3,1]))
        self.gazebo_vIMU = transform_position_I(gazebo_v_imu_ENU.reshape([3,1]))
        self.gazebo_qIMU = transform_orientation_I(transform_orientation_B((Rotation.from_quat(gazebo_q_imu_ENU)))).as_quat()
        self.gazebo_oIMU = transform_omega_B(gazebo_o_imu_ENU).reshape([3,1])

        # Convert all measurements to NED and compute auxiliar states
        
        self.pL = transform_position_I(gazebo_p_load_ENU.reshape([3,1]))
        self.vL = transform_position_I(gazebo_v_load_ENU.reshape([3,1]))

        # self.pQ = transform_position_I(p_quad_ENU.reshape([3,1]))
        # self.vQ = transform_position_I(v_quad_ENU.reshape([3,1]))

        self.pQ = transform_position_I(gazebo_p_quad_ENU.reshape([3,1]))
        self.vQ = transform_position_I(gazebo_v_quad_ENU.reshape([3,1]))

        self.q = (self.pL - self.pQ) / norm(self.pL - self.pQ)
        self.dq = (np.eye(3) - self.q @ self.q.T) @ (self.vL - self.vQ) / norm(self.pL - self.pQ)
        self.o = skew(self.q) @ self.dq 

        # self.qQ = transform_orientation_I(transform_orientation_B((Rotation.from_quat(q_quad_ENU)))).as_quat()
        # self.RQ = transform_orientation_I(transform_orientation_B((Rotation.from_dcm(R_quad_ENU)))).as_dcm()
        # self.oQ = transform_omega_B(o_quad_ENU).reshape([3,1])

        self.qQ = transform_orientation_I(transform_orientation_B((Rotation.from_quat(gazebo_q_quad_ENU)))).as_quat()
        self.RQ = transform_orientation_I(transform_orientation_B((Rotation.from_dcm(gazebo_R_quad_ENU)))).as_dcm()
        self.oQ = transform_omega_B(gazebo_o_quad_ENU).reshape([3,1])
        
        self.qL = transform_orientation_I(transform_orientation_B((Rotation.from_quat(gazebo_q_load_ENU)))).as_quat()
        self.RL = transform_orientation_I(transform_orientation_B((Rotation.from_dcm(gazebo_R_load_ENU)))).as_dcm()
        self.oL = transform_omega_B(gazebo_o_load_ENU).reshape([3,1])

        # print(f'length (ROS) = {norm(self.pL - self.pQ)}')
        # print(f'length (Gazebo) = {norm(self.gazebo_pL - self.gazebo_pQ)}')

        return

    def compare_gazebo_px4(self):
        # rospy.loginfo("Compare gazebo states and MAVROS measurements")
        rospy.loginfo(f'pQ: {(self.pQ - self.gazebo_pQ).T}')
        rospy.loginfo(f'vQ: {(self.vQ - self.gazebo_vQ).T}')
        rospy.loginfo(f'qQ: {(Rotation.from_quat(self.qQ).inv() * Rotation.from_quat(self.gazebo_qQ)).as_quat().T}')
        rospy.loginfo(f'oQ: {(self.oQ - self.gazebo_oQ).T}')

        rospy.loginfo(f'pIMU: {(self.pQ - self.gazebo_pIMU).T}')
        rospy.loginfo(f'vIMU: {(self.vQ - self.gazebo_vIMU).T}')
        rospy.loginfo(f'qIMU: {(Rotation.from_quat(self.qQ).inv() * Rotation.from_quat(self.gazebo_qIMU)).as_quat().T}')
        rospy.loginfo(f'oIMU: {(self.oQ - self.gazebo_oIMU).T}')

        rospy.loginfo(f'pL: {(self.pL - self.gazebo_pL).T}')
        rospy.loginfo(f'vL: {(self.vL - self.gazebo_vL).T}')
        rospy.loginfo(f'qL: {(Rotation.from_quat(self.qL).inv() * Rotation.from_quat(self.gazebo_qL)).as_quat().T}')
        rospy.loginfo(f'oL: {(self.oL - self.gazebo_oL).T}')
        return

    def controller_load(self):

        self.RL = np.eye(3)
        self.oL = np.zeros([3,1])
        self.delta_TLd = np.zeros(n_cables*n_dims)
        self.V = np.zeros([1,1])

        t = rospy.get_time() - self.t0
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
        e3 = np.array([0,0,1]).reshape([3,1])
        r3 = self.RQ @ e3 
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
            dy, state = process_state(0,transpose(y))
        except Exception as e:
            print("Couldn't compute state")
            print(e)
            return

        
        # print(f"t = {self.local_position.header.stamp.secs}.{self.local_position.header.stamp.nsecs}")

        # # All the same for single-quadrotor
        # print(f"state.qFd = {state.qFd.ravel()}")
        # print(f"pd     = {state.pd.ravel()}")
        # print(f"state.dpd = {state.dpd.ravel()}")
        # print(f"state.d2pd = {state.d2pd.ravel()}")
        # print(f"Fd_L   = {state.Fd.ravel()}")
        # print(f"Fd_Q   = {state.u.ravel()}")

        r3_d = (- state.u / norm(state.u)).reshape(3,1)
        # print(f"r3_Q   = {r3_d.ravel()}")

        # print(f"r3 err = {180/np.pi * np.arccos(r3.T @ r3_d)}")

        qd = state.qd.reshape(3,1)
        # print(f"qd     = {state.qd.ravel()}")
        # print(f"qd err = {180/np.pi * np.arccos(self.q.T @ qd)}")
        # print("")
        # print("")
        # print(f"state.u = {state.u.ravel()}")
      

        # print(f"q      = {self.q.ravel()}")
        # print(f"qd     = {state.qd.ravel()}")
        # print(f"qd err = {180/np.pi * np.arccos(self.q.T @ qd)}")

        # print(f"r3     = {r3.ravel()}")
        # print(f"r3_Q   = {r3_d.ravel()}")
        # print(f"r3 err = {180/np.pi * np.arccos(r3.T @ r3_d)}")

        # print("ep = ", (self.pL - state.pd))
        # print("ev = ", self.vL)
        # print("Fd = ", state.Fd)
        # print("-----------------")

        R = F_vector_to_Rotation(state.u)
        
        # print(f"R * norm(Fd) * (-e3)  = {(R).as_dcm() @ (-e3) * norm(state.Fd)}")

        kT = 1.0

        q = R.as_quat()
        T = kT*(norm(state.u)/((mass_quad+mL)*g) - 1.0) + 0.625

        T = np.clip(T,0,1)

        # print(f"q_input = {q}")
        # print(f"T_input = {T}")
        # print("")
        # print("")

        yaw = deg2rad(0)
        R_yaw = Rotation.from_rotvec(yaw*np.array([0,0,1]))
        
        # YRP = Rotation.from_quat(self.qQ).as_euler('zyx')
        # R_yaw = Rotation.from_rotvec(YRP[0]*np.array([0,0,1]))
        
        q = (R*R_yaw).as_quat()
        # q_ENU = transform_orientation_I(transform_orientation_B(R*R_yaw)).as_quat()
        
        # q_input = q
        # q_input = Rotation.from_rotvec(yaw*np.array([0,0,1])).as_quat()

        # print(f"yaw = {rad2deg(yaw)}")
        # print(f"q_input = {q}")
        # print(f"T_input = {T}")
        # print(f"state.Fd = {state.Fd}")

        # print(f"R_input * norm(Fd) * (-e3)  = {(R*R_yaw).as_dcm() @ (-e3) * norm(state.Fd)}")
        

        # output = concatenate([
        #                     state.qFd.ravel(),
        #                     state.Fd.ravel(),
        #                     0*state.Fd.ravel(),
        #                     state.I_TLd.ravel(),
        #                     state.qd.ravel(),
        #                     state.oqd.ravel(),
        #                     state.zoq.ravel(),
        #                     state.u_parallel.ravel(),
        #                     state.u_perp.ravel(),
        #                     state.e_q.ravel(),
        #                     state.e_oq.ravel(),
        #                     ])

        return (T, q)


    def controller_attitude(self):

        e3 = np.array([0,0,1]).reshape([3,1])

        kp = np.array([1,1,0.5]).reshape([3,1])
        kv = np.array([1,1,0.5]).reshape([3,1])

        # PD controller for position

        Fd = - kp*(self.pQ - self.pd) - kv*(self.vQ - self.Dpd) + mass_quad*self.D2pd - mass_quad*g*e3

        R = F_vector_to_Rotation(Fd)
        
        # print(f"R * norm(Fd) * (-e3)  = {(R).as_dcm() @ (-e3) * norm(state.Fd)}")

        kT = 1.0

        q = R.as_quat()
        T = kT*(norm(Fd)/(mass_quad*g) - 1.0) + 0.625

        T = np.clip(T,0,1)

        # print(f"q_input = {q}")
        # print(f"T_input = {T}")

        yaw = deg2rad(0)
        R_yaw = Rotation.from_rotvec(yaw*np.array([0,0,1]))
        
        # YRP = Rotation.from_quat(self.qQ).as_euler('zyx')
        # R_yaw = Rotation.from_rotvec(YRP[0]*np.array([0,0,1]))
        
        q = (R*R_yaw).as_quat()
    
        return (T, q)


    def controller_position(self):
        return self.pd


from time import sleep
        
if __name__ == '__main__':
    rospy.init_node('pyquad', anonymous=True)
    quad = MavrosQuad()

    print("Waiting for topics...")
    quad.wait_for_topics(30)
    
    quad.computeSystemStates()
    
    print("Set mode OFFBOARD...")
    quad.set_mode("OFFBOARD", 5)

    print("ARM...")
    quad.set_arm(True, 5)

    # for i in range(1000):
    #     print(quad.sub_topics_ready)
    #     quad.computeSystemStates()
    #     quad.log_topic_vars()
    #     quad.log_states()
        
    #     print(f"qQ = {quad.qQ}")
    #     print(f"                                                    pQ = {quad.pQ.T}")

    #     print(f"q_quad_ENU = {quad.q_quad_ENU}")

    #     print(f"q_input = {quad.q_input}")
    #     print(f"T_input = {quad.T_input}")

    #     sleep(1)

    while True:

        print("Now listening for options (1,2,3) - 0 to exit:")
        try:
            c = input()[0]
        except:
            continue

        if(c=='1'):
            print("* Position control *")
            quad.offboard_position_active = True
            quad.offboard_attitude_active = False
            quad.offboard_load_active = False
        if(c=='2'):
            print("* Attitude control *")
            quad.offboard_position_active = False
            quad.offboard_attitude_active = True
            quad.offboard_load_active = False

        if(c=='3'):
            print("* Load control *")
            quad.t0 = rospy.get_time()
            quad.t_prev = -0.01
            quad.offboard_position_active = False
            quad.offboard_attitude_active = False
            quad.offboard_load_active = True

        if(c=='0'):
            break
