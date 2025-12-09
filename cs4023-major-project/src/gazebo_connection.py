#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState


class GazeboConnection:

    def __init__(self):

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

    def pauseSim(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException, e):
            print("/gazebo/pause_physics service call failed")

    def unpauseSim(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException, e):
            print("/gazebo/unpause_physics service call failed")

    def resetSim(self):
        state_msg = ModelState()
        state_msg.model_name = 'mobile_base'
        state_msg.pose.position.x = 0.000664
        state_msg.pose.position.y = 0.000173
        state_msg.pose.position.z = -0.000342
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except (rospy.ServiceException, e):
            print ("Service call failed: %s" % e)
        # rospy.wait_for_service("/gazebo/reset_simulation")
        # try:
        #     self.reset_proxy()
        # except (rospy.ServiceException, e):
        #     print("/gazebo/reset_simulation service call failed")


