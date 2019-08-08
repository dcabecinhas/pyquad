#!/usr/bin/env python3

# Based on https://github.com/PX4/Firmware/blob/master/integrationtests/python_src/px4_it/mavros/mavros_test_common.py

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
            if all(value for value in self.sub_topics_ready.values()):
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
        rospy.loginfo("========================")
        rospy.loginfo(f"pL: {self.pL.T}")
        rospy.loginfo(f"vL: {self.vL.T}")
        rospy.loginfo(f"qL: {self.qL.T}")
        rospy.loginfo(f"oL: {self.oL.T}")
        # rospy.loginfo(f"R: {self.R}")
        rospy.loginfo(f"R (in RPY): {(Rotation.from_dcm(self.RQ)).as_euler('xyz')}")
        rospy.loginfo(f"R (in quat): {(Rotation.from_dcm(self.RQ)).as_quat()}")
        rospy.loginfo(f"o: {self.oQ.T}")
        rospy.loginfo(f"R_imu (in RPY): {(Rotation.from_dcm(self.R_imu)).as_euler('xyz')}")
        rospy.loginfo(f"R_imu (in quat): {(Rotation.from_dcm(self.R_imu)).as_quat()}")
        rospy.loginfo(f"o_imu: {self.o_imu.T}")
        rospy.loginfo(f"pQ: {self.pQ.T}")
        rospy.loginfo(f"vQ: {self.vQ.T}")
        
        rospy.loginfo("========================")

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
        
        self.pL = transform_position_I(p_load_ENU.reshape([3,1]))
        self.vL = transform_position_I(v_load_ENU.reshape([3,1]))

        self.pQ = transform_position_I(p_quad_ENU.reshape([3,1]))
        self.vQ = transform_position_I(v_quad_ENU.reshape([3,1]))

        self.q = (self.pL - self.pQ) / norm(self.pL - self.pQ)
        self.o = self.q @ self.q.T @ (self.vL - self.vQ) / norm(self.pL - self.pQ)
        
        self.qQ = transform_orientation_I(transform_orientation_B((Rotation.from_quat(q_quad_ENU)))).as_quat()
        self.RQ = transform_orientation_I(transform_orientation_B((Rotation.from_dcm(R_quad_ENU)))).as_dcm()
        self.oQ = transform_omega_B(o_quad_ENU).reshape([3,1])
        
        self.qL = transform_orientation_I(transform_orientation_B((Rotation.from_quat(gazebo_q_load_ENU)))).as_quat()
        self.RL = transform_orientation_I(transform_orientation_B((Rotation.from_dcm(gazebo_R_load_ENU)))).as_dcm()
        self.oL = transform_omega_B(gazebo_o_load_ENU).reshape([3,1])

        return

    def compare_gazebo_px4(self):
        rospy.loginfo("Compare gazebo states and MAVROS measurements")
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


if __name__ == '__main__':
    rospy.init_node('pyquad', anonymous=True)
    quad = MavrosQuad()
    quad.wait_for_topics(10)
    # quad.computeSystemStates()
    # quad.log_topic_vars()
    # quad.log_states()
    for i in range(5):
        quad.computeSystemStates()
        quad.log_states()
        quad.compare_gazebo_px4()