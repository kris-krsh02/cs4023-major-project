#!/usr/bin/env python3

import time, math
import numpy as np
# import tensorflow as tf
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import utils, spaces
from gymnasium.utils import seeding
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple
import rospy
import subprocess
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from kobuki_msgs.msg import BumperEvent
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from gazebo_connection import GazeboConnection
from scipy.spatial.transform import Rotation


reg = register(
    id="Turtlebot3Touring",
    entry_point="touring_env:TurtlebotTouringEnv",
    # timestep_limit=300,  # TODO: timestep might be too big/too little
)


class TurtlebotTouringEnv(gym.Env):

    def __init__(self):
        self.vel_pub = rospy.Publisher(
            "/cmd_vel_mux/input/navi", Twist, queue_size=1
        )  # TODO: Revisit queue size

        # TODO: How to implement and enumerate Pose goals here?
        # need to come up with the sequential logic
        self.curr_goal = Point(2, 2, 0)  # TODO: Define accurate
        self.scan_angles = []

        # Velocity parameters
        self.linear_velocity = rospy.get_param("/linear_velocity")
        self.fr_angular = rospy.get_param("/fast_right_angular_velocity")
        self.fl_angular = rospy.get_param("/fast_left_angular_velocity")
        self.sr_angular = rospy.get_param("/slow_right_angular_velocity")
        self.sl_angular = rospy.get_param("/slow_left_angular_velocity")

        self.running_step = rospy.get_param("/running_step")
        self.goal_threshold = rospy.get_param("/goal_thres")
        self.obstacle_threshold = rospy.get_param("/obstacle_thres")

        # Connect with gazebo
        self.gazebo = GazeboConnection()

        # Initialize action space: 5 angular velocities with constant linear velocity
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple((Box(low=0,high=10,shape=(43,),dtype=np.float32), Discrete(2), Box(low=0,high=10,dtype=np.float32), Box(low=0,high=360)))

        # No reward bounds
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def _seed(self, seed=None):
        """
        Docstring for _seed

        :param self: Description
        :param seed: Description
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Docstring for _reset

        :param self: Description
        """

        # output = subprocess.run(["rosnode", "kill", "/depthimage_to_laserscan"], capture_output=True, text=True)
        # print(output)
        # output = subprocess.run(["rosnode", "kill", "/laserscan_nodelet_manager"], capture_output=True, text=True)
        # print(output)

        self.gazebo.resetSim()
        # output = subprocess.run(["roslaunch", "cs4023-major-project", "fake_scanner.launch"], capture_output=True, text=True)
        # print(output)
        self.gazebo.unpauseSim()

        # TODO: Reset robot conditions
        self.reset_robot()
        data = self.take_observation()
        self.gazebo.pauseSim()
        return data, {}

    def step(self, action):
        """
        Docstring for _step

        :param self: Description
        :param action: Description
        """

        vel_cmd = Twist()
        if action == 0:  # Fast Left
            vel_cmd.linear.x = self.linear_velocity
            vel_cmd.angular.z = self.fl_angular
        elif action == 1:  # Slow Left
            vel_cmd.linear.x = self.linear_velocity
            vel_cmd.angular.z = self.sl_angular
        elif action == 2:  # Forward
            vel_cmd.linear.x = self.linear_velocity
            vel_cmd.angular.z = 0.0
        elif action == 3:  # Slow Right
            vel_cmd.linear.x = self.linear_velocity
            vel_cmd.angular.z = self.sr_angular
        elif action == 4:  # Fast Right
            vel_cmd.linear.x = self.linear_velocity
            vel_cmd.angular.z = self.fr_angular

        self.gazebo.unpauseSim()
        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)
        data = self.take_observation()
        self.gazebo.pauseSim()

        reward, done = self.calculate_reward(data)
        return data, reward, done, {}

    def reset_robot(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

    def take_observation(self):
        # LaserScan data
        scan_ranges = None
        while scan_ranges is None:
            try:
                scan_ranges = rospy.wait_for_message("/scan", LaserScan, timeout=5)
            except:
                rospy.loginfo("LaserScan has no data, retrying...")

        # Bumper data
        collision = 0
        try:
            collision = rospy.wait_for_message(
                "/mobile_base/events/bumper", BumperEvent, timeout=0.5
            ).state
            if collision == 0:
                collision = 1
        except:
            pass
            # rospy.loginfo("No collision detected")

        # Distance to goal and Heading angle
        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message("/odom", Odometry, timeout=5).pose
            except:
                rospy.loginfo("Odometer has no data, retrying...")

        scan_ranges = self.process_scan_ranges(scan_ranges)
        distance = self.calculate_distance(pose.pose.position, self.curr_goal)
        heading_angle = self.calculate_heading_angle(pose)

        return (scan_ranges, collision, distance, heading_angle)

    def process_scan_ranges(self, scan_ranges):
        filtered_scan_ranges = []
        scan_angles = []

        for i in range(0, len(scan_ranges.ranges), 15):
            if math.isnan(scan_ranges.ranges[i]):
                filtered_scan_ranges.append(10.0)
            else:
                filtered_scan_ranges.append(scan_ranges.ranges[i])
            angle = scan_ranges.angle_min + i * scan_ranges.angle_increment
            scan_angles.append(angle)
        self.scan_angles = scan_angles
        return filtered_scan_ranges

    def calculate_reward(self, data: tuple):
        """
        Docstring for calculate_reward

        :param self: Description
        :param data: scan_range, collision, distance, heading_angle
        """
        done = False

        if data[1]:  # collision detection
            done = True
            reward = -50.0

        elif data[2] <= self.goal_threshold:  # within range of goal
            done = True
            reward = 100.0

        else:
            angle_reward = 1 - (2 * abs(data[3]) / math.pi)
            obstacle_reward = self.calculate_weighted_obstacle_reward(data[0], self.scan_angles)
            reward = angle_reward + obstacle_reward

        return reward, done

    def calculate_weighted_obstacle_reward(self, scan_ranges, scan_angles):
        scan_ranges = np.array(scan_ranges)
        scan_angles = np.array(scan_angles)
        valid_mask = scan_ranges <= 0.5
        if not np.any(valid_mask):
            return 0.0

        scan_ranges = scan_ranges[valid_mask]
        scan_angles = scan_angles[valid_mask]

        relative_angles = np.unwrap(scan_angles)
        relative_angles[relative_angles > np.pi] -= 2 * np.pi

        weights = self.compute_directional_weights(relative_angles, max_weight=10.0)
        safe_dists = np.clip(scan_ranges - 0.25, 1e-2, 3.5)
        decay = np.exp(-3.0 * safe_dists)
        weighted_decay = np.dot(weights, decay)

        reward = -(1.0 + 4.0 * weighted_decay)
        return reward

    def compute_directional_weights(self, relative_angles, max_weight=10.0):
        power = 6
        raw_weights = (np.cos(relative_angles)) ** power + 0.1
        scaled_weights = raw_weights * (max_weight / np.max(raw_weights))
        normalized_weights = scaled_weights / np.sum(scaled_weights)
        return normalized_weights

    def calculate_distance(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def calculate_heading_angle(self, pose):
        path = math.atan2(
            self.curr_goal.y - pose.pose.position.y,
            self.curr_goal.x - pose.pose.position.x,
        )

        # _, _, theta = tf.transformations.euler_from_quaternion(
        #     (
        #         pose.pose.orientation.x,
        #         pose.pose.orientation.y,
        #         pose.pose.orientation.z,
        #         pose.pose.orientation.w,
        #     )
        # )

        input = Rotation.from_quat([pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w])

        _, _, theta = input.as_euler("xyz")

        heading_angle = path - theta

        if heading_angle > math.pi:
            heading_angle -= 2 * math.pi

        elif heading_angle < -math.pi:
            heading_angle += 2 * math.pi

        return heading_angle
