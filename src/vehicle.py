#!/usr/bin/env python3

# from single_quadrotor import *

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

    def __init__(self, ID=0):
        # super(Vehicle, self).__init__(*args)

        self.ID = ID
        
        self.state = State()

        # ROS services
        service_timeout = 10
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.wait_for_service(f'uav_{self.ID}/mavros/param/get', service_timeout)
            rospy.wait_for_service(f'uav_{self.ID}/mavros/cmd/arming', service_timeout)
            rospy.wait_for_service(f'uav_{self.ID}/mavros/mission/push', service_timeout)
            rospy.wait_for_service(f'uav_{self.ID}/mavros/mission/clear', service_timeout)
            rospy.wait_for_service(f'uav_{self.ID}/mavros/set_mode', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            rospy.logerr("failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy(f'uav_{self.ID}/mavros/param/get', ParamGet)
        self.set_arming_srv = rospy.ServiceProxy(f'uav_{self.ID}/mavros/cmd/arming',
                                                 CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(f'uav_{self.ID}/mavros/set_mode', SetMode)
        self.wp_clear_srv = rospy.ServiceProxy(f'uav_{self.ID}/mavros/mission/clear',
                                               WaypointClear)
        self.wp_push_srv = rospy.ServiceProxy(f'uav_{self.ID}/mavros/mission/push',
                                              WaypointPush)

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

    def assertTrue(self, result, text):
        if(not result):
            rospy.loginfo(text)

from time import sleep
        
if __name__ == '__main__':
    rospy.init_node('pyquad', anonymous=True)
    
    number_of_vehicles = 4

    quad_list = []
    for i in range(number_of_vehicles):
        
        print(f"[{i}] Start vehicle object...")
        quad = Vehicle(ID=i)
        quad_list.append(quad)

    for i in range(number_of_vehicles):
        # Run each in a thread
        print(f"[{i}] Set mode OFFBOARD...")
        quad_list[i].set_mode("OFFBOARD", 5)

    for i in range(number_of_vehicles):
        print(f"[{i}] ARM...")
        quad_list[i].set_arm(True, 5)

    while True:

        print("Now listening for options - 0 to exit:")
        try:
            c = input()[0]
        except:
            continue

        # if(c=='1'):
        #     print("* Position control *")
        #     quad.offboard_position_active = True
        #     quad.offboard_attitude_active = False
        #     quad.offboard_load_active = False
        # if(c=='2'):
        #     print("* Attitude control *")
        #     quad.offboard_position_active = False
        #     quad.offboard_attitude_active = True
        #     quad.offboard_load_active = False

        # if(c=='3'):
        #     print("* Load control *")
        #     quad.t0 = rospy.get_time()
        #     quad.t_prev = -0.01
        #     quad.offboard_position_active = False
        #     quad.offboard_attitude_active = False
        #     quad.offboard_load_active = True

        if(c=='0'):
            break
