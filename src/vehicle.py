#!/usr/bin/env python3

# from single_quadrotor import *

from threading import Thread
from time import sleep

import rospy
# import math

# import numpy as np
# from scipy.spatial.transform import Rotation
# from scipy.linalg import norm

# from gazebo_msgs.msg import LinkStates
# from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Twist, Quaternion, Vector3
# from sensor_msgs.msg import BatteryState, Imu, NavSatFix
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, \
                            WaypointPush

from mavros.utils import *

from pymavlink import mavutil

# from frame_conversions import *

# from std_msgs.msg import Header
# from threading import Thread

# np.set_printoptions(precision=4)
# np.set_printoptions(suppress=True)
# np.set_printoptions(floatmode="fixed")
# np.set_printoptions(sign=" ")

class Vehicle():

    def __init__(self, ID=0, vehicle_cfg=[]):
        # super(Vehicle, self).__init__(*args)

        self.ID = ID
        
        self.state = State()

        print("Vehicle: ", vehicle_cfg['mavros'])
        
        # ROS services
        service_timeout = 10
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.wait_for_service(vehicle_cfg['mavros'] + '/mavros/param/get', service_timeout)
            rospy.wait_for_service(vehicle_cfg['mavros'] + '/mavros/cmd/arming', service_timeout)
            rospy.wait_for_service(vehicle_cfg['mavros'] + '/mavros/mission/push', service_timeout)
            rospy.wait_for_service(vehicle_cfg['mavros'] + '/mavros/mission/clear', service_timeout)
            rospy.wait_for_service(vehicle_cfg['mavros'] + '/mavros/set_mode', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            rospy.logerr("failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy(vehicle_cfg['mavros'] + '/mavros/param/get', ParamGet)
        self.set_arming_srv = rospy.ServiceProxy(vehicle_cfg['mavros'] + '/mavros/cmd/arming',
                                                 CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(vehicle_cfg['mavros'] + '/mavros/set_mode', SetMode)
        self.wp_clear_srv = rospy.ServiceProxy(vehicle_cfg['mavros'] + '/mavros/mission/clear',
                                               WaypointClear)
        self.wp_push_srv = rospy.ServiceProxy(vehicle_cfg['mavros'] + '/mavros/mission/push',
                                              WaypointPush)

        # self.state_sub = rospy.Subscriber(vehicle_cfg['mavros'] + '/mavros/state', 
        #                                             State,
        #                                             self.state_callback)

        # # ROS subscribers
        # self.sub_topics_ready = {
        #     key: False
        #     for key in [
        #         'state'
        #     ]
        # }
        
        # send setpoints in seperate thread to better prevent failsafe
        self.thread = Thread(target=self.activate, args=())
        self.thread.daemon = True
        # self.thread.start()


    #
    # Callback functions
    #
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
            
    #
    # Helper methods
    #
    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 10  # Hz
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
                        print(res)
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
        loop_freq = 10  # Hz
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

    def assertTrue(self, result, text):
        if(not result):
            rospy.loginfo(text)

    def wait_for_topics(self, timeout):
            """wait for simulation to be ready, make sure we're getting topic info
            from all topics by checking dictionary of flag values set in callbacks,
            timeout(int): seconds"""
            rospy.loginfo("waiting for subscribed topics to be ready")
            loop_freq = 10  # Hz
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
                
    def activate(self):
        self.set_mode("OFFBOARD", 3)
        self.set_arm(True, 3)

if __name__ == '__main__':
    pass

    # rospy.init_node('pyquad', anonymous=True)
    
    # number_of_vehicles = 1

    # quad_list = []
    # for i in range(number_of_vehicles):
    #     print(f"[{i}] Start vehicle object...")
    #     quad = Vehicle(ID=i)

    #     # print(i, '- a')
    #     quad.wait_for_topics(10)
    #     # print(i, '- b')
    #     quad.thread.start()
    #     # print(i, '- c')
    #     quad_list.append(quad)
        
    # for i in range(number_of_vehicles):
    #     # Run each in a thread
    #     print(f"[{i}] Set mode OFFBOARD...")
    #     quad_list[i].set_mode("OFFBOARD", 5)

    # for i in range(number_of_vehicles):
    #     print(f"[{i}] ARM...")
    #     quad_list[i].set_arm(True, 5)

    # while True:

    #     print("Now listening for options - 0 to exit:")
    #     try:
    #         c = input()[0]
    #     except:
    #         continue

    #     # if(c=='1'):
    #     #     print("* Position control *")
    #     #     quad.offboard_position_active = True
    #     #     quad.offboard_attitude_active = False
    #     #     quad.offboard_load_active = False
    #     # if(c=='2'):
    #     #     print("* Attitude control *")
    #     #     quad.offboard_position_active = False
    #     #     quad.offboard_attitude_active = True
    #     #     quad.offboard_load_active = False

    #     # if(c=='3'):
    #     #     print("* Load control *")
    #     #     quad.t0 = rospy.get_time()
    #     #     quad.t_prev = -0.01
    #     #     quad.offboard_position_active = False
    #     #     quad.offboard_attitude_active = False
    #     #     quad.offboard_load_active = True

    #     if(c=='0'):
    #         break
